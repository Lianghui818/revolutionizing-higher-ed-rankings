import requests
import pandas as pd
import time

# Function to fetch papers from DBLP API
def fetch_dblp_papers(query, start=0, max_results=1000, format="json"):
    url = "https://dblp.org/search/publ/api"
    params = {
        "q": query,
        "h": max_results,       # Number of results to fetch
        "f": start,             # Starting point since max to 1000 once
        "format": format
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to parse the DBLP response
def parse_dblp_response(data):
    papers = []
    if "result" in data and "hits" in data["result"]:
        hits = data["result"]["hits"]
        if "hit" in hits:
            for hit in hits["hit"]:
                info = hit.get("info", {})
                authors_data = info.get("authors", {}).get("author", [])
                if isinstance(authors_data, dict):
                    authors = [authors_data.get("text", "Unknown")]
                elif isinstance(authors_data, list):
                    authors = [author.get("text", "Unknown") for author in authors_data]
                else:
                    authors = ["Unknown"]
                
                # Count number of authors
                author_count = len(authors)
                
                papers.append({
                    "Title": info.get("title", "N/A"),
                    "Authors": ", ".join(authors),
                    "Author_Count": author_count,  # Added author count
                    "Venue": info.get("venue", "N/A"),
                    "Year": info.get("year", "N/A"),
                    "URL": info.get("ee", "N/A"),
                })
    return papers

# Save papers to a CSV file
def save_to_csv(papers, filename):
    df = pd.DataFrame(papers)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(papers)} papers to {filename}")

# Fetch and save papers
def main():
    query = "ICCV 2023"                 # Replace with other conference: need to check with Fuxin
    all_papers = []
    start = 0
    max_results = 1000
    total_fetched = 0

    while True:
        print(f"Fetching papers from DBLP... Start at {start}")
        response = fetch_dblp_papers(query, start=start, max_results=max_results)
        if response:
            papers = parse_dblp_response(response)
            if not papers:
                print("No more papers found.")
                break
            all_papers.extend(papers)
            total_fetched += len(papers)
            print(f"Fetched {len(papers)} papers. Total fetched: {total_fetched}")
            start += max_results
            time.sleep(1)           # Pause to avoid overwhelming the server
        else:
            print("Error fetching papers.")
            break
        
    if all_papers:
        filename = f"{query.replace(' ', '_')}_papers.csv"
        save_to_csv(all_papers, filename)
    else:
        print("No papers found!")


if __name__ == "__main__":
    main()
