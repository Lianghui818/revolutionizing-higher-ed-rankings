import json
import csv
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

INPUT_JSON = "/nfs/stak/users/wangl9/hpc-share/revolutionizing-higher-ed-rankings/AT/ICCV2023_1006_re.json_partial.json"
OUTPUT_CSV = "ICCV2023_1006_full_names.csv"


def fetch_dblp_authors_and_title(title, abbreviated_authors):
    """Fetch disambiguated full author names from DBLP using the given paper title."""
    search_url = f"https://dblp.org/search?q={urllib.parse.quote(title)}"

    # Setup retry session
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=10,  # Waits: 10s, 20s, 40s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(search_url)
    except requests.exceptions.RequestException as e:
        print(f"Request failed for title '{title}': {e}")
        return []

    if response.status_code != 200:
        print(f"Failed to fetch DBLP data for: {title} (status {response.status_code})")
        return []

    print(f"Fetched DBLP data for: {title}")
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("cite", class_="data tts-content")

    for result in results:
        author_spans = result.find_all("span", itemprop="name")
        full_authors = [
            span.get("title", span.get_text().strip()).strip()
            for span in author_spans[:-2]
        ]

        print(f"Full authors found: {full_authors}")

        for abbrev_author in abbreviated_authors:
            if any(abbrev_author.split()[-1].strip("'") in full_name for full_name in full_authors):
                return full_authors

    print(f"No exact match found for: {title}")
    return []

def process_json(input_json, output_csv):
    existing_entries = set()
    write_header = not os.path.exists(output_csv)

    # Read existing output to prevent duplicates
    try:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    existing_entries.add(row[0])
    except FileNotFoundError:
        pass
    print(f"Existing entries loaded: {existing_entries}")

    with open(output_csv, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["Full Name", "Normalized Score"])

        with open(input_json, "r", encoding="utf-8") as f:
            papers = json.load(f)
            total_papers = len(papers)

            for idx, paper in enumerate(papers, 1):
                print(f"\n[Progress] Processing paper {idx}/{total_papers}")

                # FIX: Check if paper is None before accessing its attributes
                if paper is None:
                    print(f"Skipping None entry at index {idx}")
                    continue

                # FIX: Also check if paper is not a dictionary
                if not isinstance(paper, dict):
                    print(f"Skipping non-dictionary entry at index {idx}: {type(paper)}")
                    continue

                title = paper.get("title", "")
                if title is None:  # Additional safety check
                    title = ""
                
                # Handle case where title is a list instead of string
                if isinstance(title, list):
                    if len(title) > 0:
                        title = str(title[0])  # Take first element and convert to string
                    else:
                        title = ""  # Empty list becomes empty string
                
                # Ensure title is a string before calling strip()
                title = str(title).strip()
                
                authors = paper.get("authors", [])
                if authors is None:  # Additional safety check
                    authors = []
                
                original_authors = authors

                if not title or not authors or not original_authors:
                    print(f"Skipping entry {idx} due to missing title or authors")
                    continue

                # Check if abbreviated (presence of ".")
                if any("." in a for a in authors if a is not None):
                    print(f"Processing (abbreviated): {title}")
                    full_names = fetch_dblp_authors_and_title(title, authors)
                    if full_names:
                        for name in full_names:
                            if name not in existing_entries:
                                writer.writerow([name, 1 / (len(original_authors) * len(authors))])
                                existing_entries.add(name)
                    else:
                        print(f"Could not find full names for: {title}")
                    time.sleep(random.uniform(15, 30))
                else:
                    for a in authors:
                        if a is None:  # Skip None author names
                            continue
                        name = a.strip()
                        if name and name not in existing_entries:
                            writer.writerow([name, 1 / (len(original_authors) * len(authors))])
                            existing_entries.add(name)
                        else:
                            print(f"Skipping duplicate or empty name: {name}")

    print("Processing complete. Results saved.")

def main():
    process_json(INPUT_JSON, OUTPUT_CSV)

if __name__ == "__main__":
    main()