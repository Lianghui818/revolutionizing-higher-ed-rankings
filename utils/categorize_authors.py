import pandas as pd
from pathlib import Path
import requests
from io import StringIO
import string
import re
import json
from datetime import datetime

CONFERENCE_CATEGORIES = {
    'AAAI': 'Artificial Intelligence & Machine Learning', 'IJCAI': 'Artificial Intelligence & Machine Learning',
    'NeurIPS': 'Artificial Intelligence & Machine Learning', 'ICML': 'Artificial Intelligence & Machine Learning',
    'KDD': 'Data Science & Data Mining', 'ICDM': 'Data Science & Data Mining', 'SDM': 'Data Science & Data Mining',
    'CIKM': 'Data Science & Data Mining', 'WWW': 'Data Science & Data Mining',
    'CVPR': 'Computer Vision & Image Processing', 'ICCV': 'Computer Vision & Image Processing',
    'ECCV': 'Computer Vision & Image Processing',
    'ACL': 'Natural Language Processing', 'EMNLP': 'Natural Language Processing',
    'NAACL': 'Natural Language Processing', 'COLING': 'Natural Language Processing',
    'SOSP': 'Systems & Networking', 'OSDI': 'Systems & Networking', 'SIGCOMM': 'Systems & Networking',
    'NSDI': 'Systems & Networking', 'PODC': 'Systems & Networking', 'DISC': 'Systems & Networking',
    'FAST': 'Systems & Networking',
    'SIGMOD': 'Databases', 'VLDB': 'Databases', 'ICDE': 'Databases', 'EDBT': 'Databases',
    'CCS': 'Security & Privacy', 'USENIX Security': 'Security & Privacy', 'NDSS': 'Security & Privacy',
    'S&P (Oakland)': 'Security & Privacy',
    'CHI': 'Human Computer Interaction', 'UIST': 'Human Computer Interaction',
    'STOC': 'Theoretical Computer Science', 'FOCS': 'Theoretical Computer Science',
    'ICALP': 'Theoretical Computer Science', 'SODA': 'Theoretical Computer Science',
    'ICSE': 'Software Engineering', 'FSE': 'Software Engineering',
    'ASE': 'Software Engineering', 'ISSTA': 'Software Engineering',
    'SIGGRAPH': 'Computer Graphics & Virtual Reality', 'Eurographics': 'Computer Graphics & Virtual Reality',
    'IEEE VR': 'Computer Graphics & Virtual Reality',
    'QIP': 'Quantum Computing', 'TQC': 'Quantum Computing',
    'CoRL': 'Interdisciplinary Fields', 'AMIA': 'Interdisciplinary Fields', 'L@S': 'Interdisciplinary Fields',
}


CONFIG_FILE = "conference_mappings.json"

USE_LOCAL_CSRANKINGS = False
CSRANKINGS_BASE_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages"


def load_saved_mappings():
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_mappings(mappings):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved mappings to {CONFIG_FILE}")


def smart_extract_conference(filename):
    base_name = Path(filename).stem
    
    
    patterns = [
        (r'^(CVPR|ICCV|ECCV|AAAI|IJCAI|ICML|NEURIPS|NeurIPS|ACL|EMNLP)\d{4}', 1),
        (r'^(CVPR|ICCV|ECCV|AAAI|IJCAI|ICML|NEURIPS|NeurIPS|ACL|EMNLP)', 1),
        (r'(CVPR|ICCV|ECCV|AAAI|IJCAI|ICML|NEURIPS|NeurIPS|ACL|EMNLP)', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, base_name, re.IGNORECASE)
        if match:
            conference = match.group(group).upper()
            if conference == 'NEURIPS':
                conference = 'NeurIPS'
            if conference in CONFERENCE_CATEGORIES:
                return conference
    
    base_upper = base_name.upper()
    for conf in CONFERENCE_CATEGORIES.keys():
        if conf.upper() in base_upper:
            return conf
    
    return None

def analyze_and_suggest_mappings(faculty_folder):
    faculty_folder = Path(faculty_folder)
    csv_files = list(faculty_folder.glob("*.csv"))
    
    suggestions = {}
    file_info = []
    
    print(f"\n🔍 ANALYZING {len(csv_files)} CSV FILES...")
    print("="*80)
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                row_count = sum(1 for line in f) - 1

            suggested_conf = smart_extract_conference(csv_file.name)
            
            info = {
                'filename': csv_file.name,
                'rows': row_count,
                'suggested_conference': suggested_conf,
                'confidence': 'High' if suggested_conf else 'Unknown'
            }
            
            file_info.append(info)
            
            print(f"{i:2d}. {csv_file.name}")
            print(f"------ Rows: {row_count:,}")
            if suggested_conf:
                category = CONFERENCE_CATEGORIES.get(suggested_conf, 'Unknown')
                print(f"    Suggested: {suggested_conf} ({category})")
                print(f"    Confidence: High")
            else:
                print(f"    Suggested: Unknown - needs manual input")
                print(f"    Confidence: Low")
            print()
            
        except Exception as e:
            print(f"{i:2d}. {csv_file.name} - ERROR: {e}")
            file_info.append({
                'filename': csv_file.name,
                'rows': 0,
                'suggested_conference': None,
                'confidence': 'Error'
            })
    
    return file_info

def interactive_conference_selection(file_info, saved_mappings):
    mappings = {}
    
    print(f"\n  INTERACTIVE CONFERENCE SELECTION")
    print("="*80)

    conf_list = sorted(CONFERENCE_CATEGORIES.keys())
    print("Available conferences:")
    for i, conf in enumerate(conf_list):
        category = CONFERENCE_CATEGORIES[conf]
        print(f"  {i+1:2d}. {conf:<12} - {category}")
    
    print(f"\n💡 Instructions:")
    print("  - Press ENTER to accept suggestion")
    print("  - Type conference name (e.g., 'CVPR')")
    print("  - Type 'skip' to skip this file")
    print("  - Type 'quit' to stop processing")
    print()
    
    for i, info in enumerate(file_info, 1):
        if info['rows'] == 0:  # Skip error files
            continue
            
        filename = info['filename']
        suggested = info['suggested_conference']
        
        if filename in saved_mappings:
            saved_conf = saved_mappings[filename]
            print(f"[{i}/{len(file_info)}] - {filename}")
            print(f"Previously saved: {saved_conf}")
            choice = input(f"    Use saved mapping? (y/n/new conference): ").strip().lower()
            if choice in ['y', 'yes', '']:
                mappings[filename] = saved_conf
                print(f"Using: {saved_conf}")
                continue
        
        print(f"[{i}/{len(file_info)}]  {filename} ({info['rows']:,} rows)")
        
        if suggested:
            category = CONFERENCE_CATEGORIES.get(suggested, '')
            prompt = f"Suggested: {suggested} ({category})\n    Your choice (ENTER to accept): "
        else:
            prompt = f"No suggestion available\n    Enter conference: "
        
        while True:
            choice = input(prompt).strip()
            
            # Accept suggestion
            if choice == '' and suggested:
                mappings[filename] = suggested
                print(f"Selected: {suggested}")
                break
            
            # Quit processing
            if choice.lower() == 'quit':
                print("Quitting...")
                return mappings
            
            # Skip file
            if choice.lower() == 'skip':
                print(f"Skipped: {filename}")
                break
            
            # Try number selection
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(conf_list):
                    selected_conf = conf_list[idx]
                    mappings[filename] = selected_conf
                    print(f"Selected: {selected_conf}")
                    break
                else:
                    print(f"Invalid number. Please enter 1-{len(conf_list)}")
                    continue
            
            # Try conference name
            choice_upper = choice.upper()
            if choice_upper in CONFERENCE_CATEGORIES:
                mappings[filename] = choice_upper
                print(f"Selected: {choice_upper}")
                break
            else:
                print(f"Unknown conference: {choice}")
                print(f"Available: {', '.join(conf_list[:5])}...")
                continue
    
    return mappings


def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df

def load_csrankings_data():
    # 加载CSrankings数据
    print("Loading CSrankings data from GitHub...")
    
    all_authors = []
    letters = string.ascii_lowercase
    
    for letter in letters:
        url = f"{CSRANKINGS_BASE_URL}/csrankings-{letter}.csv"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df = normalize_columns(df)
                all_authors.append(df)
        except Exception as e:
            print(f"[WARNING] Failed to download csrankings-{letter}.csv: {e}")
    
    if not all_authors:
        raise ValueError("No CSrankings data loaded.")
    
    combined_df = pd.concat(all_authors, ignore_index=True)
    print(f"Loaded {len(combined_df)} total authors from CSrankings")
    
    if 'Name' in combined_df.columns and 'name' not in combined_df.columns:
        combined_df['name'] = combined_df['Name']
    if 'Affiliation' in combined_df.columns and 'affiliation' not in combined_df.columns:
        combined_df['affiliation'] = combined_df['Affiliation']
    
    return combined_df

def create_author_university_mapping(csrankings_df):
    clean_df = csrankings_df.dropna(subset=['name', 'affiliation'])
    clean_df = clean_df[clean_df['name'].str.strip() != '']
    clean_df = clean_df[clean_df['affiliation'].str.strip() != '']
    
    author_univ_map = clean_df.drop_duplicates('name').set_index('name')['affiliation'].to_dict()
    print(f"[INFO] Created mapping for {len(author_univ_map)} unique authors")
    return author_univ_map

def process_single_file(csv_file, conference_name, author_univ_map):
    try:
        faculty_df = normalize_columns(pd.read_csv(csv_file))
    except Exception as e:
        print(f"[ERROR] Cannot read {csv_file}: {e}")
        return pd.DataFrame()
   
    name_column = None
    for col in faculty_df.columns:
        if any(keyword in col.lower() for keyword in ['full name', 'name', 'author', 'faculty']):
            name_column = col
            break
    
    if not name_column:
        print(f"[ERROR] No name column found in {csv_file}")
        print(f"[INFO] Available columns: {list(faculty_df.columns)}")
        return pd.DataFrame()
    
    category = CONFERENCE_CATEGORIES.get(conference_name, "Unknown")
    matched_rows = []
    total_faculty = len(faculty_df)
    matched_count = 0
    
    for _, row in faculty_df.iterrows():
        name = str(row[name_column]).strip()
        if not name or name == 'nan':
            continue
            
        university = author_univ_map.get(name)
        
        if university:
            matched_count += 1
            row["University"] = university
            row["Matched Conferences"] = conference_name
            row["Categories"] = category
            row["Full Name"] = name  
            matched_rows.append(row)
    
    match_rate = (matched_count / total_faculty * 100) if total_faculty > 0 else 0
    print(f"Processed: {matched_count:,}/{total_faculty:,} ({match_rate:.1f}%)")
    
    return pd.DataFrame(matched_rows)

def update_university_rankings(faculty_df, temp_rankings=None):
    # update ranking
    
    FIXED_COLUMNS = [
        'Index',
        'University', 
        'Artificial Intelligence & Machine Learning',
        'Data Science & Data Mining',
        'Computer Vision & Image Processing', 
        'Natural Language Processing',
        'Systems & Networking',
        'Databases',
        'Security & Privacy',
        'Human Computer Interaction',
        'Theoretical Computer Science',
        'Software Engineering',
        'Computer Graphics & Virtual Reality',
        'Quantum Computing',
        'Interdisciplinary Fields',
        'Continent'
    ]
    
    category_columns = [col for col in FIXED_COLUMNS if col not in ['Index', 'University', 'Continent']]
    
    if temp_rankings is None:
        rankings_df = pd.DataFrame(columns=FIXED_COLUMNS)
        for col in category_columns:
            rankings_df[col] = rankings_df[col].astype(float)
    else:
        rankings_df = temp_rankings.copy()
        
        for col in FIXED_COLUMNS:
            if col not in rankings_df.columns:
                if col in category_columns:
                    rankings_df[col] = 0.0
                elif col == 'Index':
                    rankings_df[col] = 0
                elif col == 'Continent':
                    rankings_df[col] = 'Unknown'
        
        for col in category_columns:
            rankings_df[col] = pd.to_numeric(rankings_df[col], errors='coerce').fillna(0.0)
    
    for _, row in faculty_df.iterrows():
        university = row["University"]
        categories = row["Categories"]
        
        try:
            score = float(row.get("Normalized Score", 1))
        except (ValueError, TypeError):
            score = 1.0
        
        university_mask = rankings_df["University"] == university
        
        if not university_mask.any():
            new_row = {}
            new_row["University"] = university
            new_row["Continent"] = "Unknown"  
            
            for cat in category_columns:
                new_row[cat] = 0.0
            
            new_row["Index"] = 0
            
            new_row_df = pd.DataFrame([new_row])
            rankings_df = pd.concat([rankings_df, new_row_df], ignore_index=True)
            
            university_mask = rankings_df["University"] == university
        
        for category in categories.split(", "):
            category = category.strip()
            if category in category_columns:
                rankings_df.loc[university_mask, category] += score
            else:
                print(f"[WARNING] Category '{category}' not found in standard columns")
    
    rankings_df = rankings_df.reindex(columns=FIXED_COLUMNS)
    
    rankings_df['Index'] = range(1, len(rankings_df) + 1)
    
    for col in category_columns:
        rankings_df[col] = rankings_df[col].astype(float)
    
    return rankings_df


def main_interactive_processor(faculty_folder, output_file):
    
    print(f"Faculty folder: {faculty_folder}")
    print(f"Output file: {output_file}")
    print(f"Config file: {CONFIG_FILE}")
    
    file_info = analyze_and_suggest_mappings(faculty_folder)
    
    saved_mappings = load_saved_mappings()
    
    print(f"\n======= PROCESSING MODE SELECTION: =======")
    print("1. Interactive mode - Confirm each file")
    print("2. Auto mode - Use suggestions + saved mappings")
    print("3. Single file mode - Process one file at a time")
    
    mode = input("Select mode (1/2/3): ").strip()
    
    if mode == '3':
        return single_file_mode(faculty_folder)
    elif mode == '2':
        mappings = {}
        for info in file_info:
            if info['suggested_conference']:
                mappings[info['filename']] = info['suggested_conference']
        mappings.update(saved_mappings)
    else:  # Default to interactive
        mappings = interactive_conference_selection(file_info, saved_mappings)
    
    if not mappings:
        print("No files selected for processing.")
        return
    
    save_mappings(mappings)
    
    try:
        csrankings_df = load_csrankings_data()
        author_univ_map = create_author_university_mapping(csrankings_df)
    except Exception as e:
        print(f"[ERROR] Failed to load CSrankings data: {e}")
        return
    
    faculty_folder = Path(faculty_folder)
    accumulated_rankings = None
    total_processed = 0
    results = []
    
    print(f"\nProcessing {len(mappings)} selected files...\n")
    
    for i, (filename, conference) in enumerate(mappings.items(), 1):
        csv_file = faculty_folder / filename
        # print(f"\n[{i}/{len(mappings)}]  {filename} → {conference}")
        print(f"\n{filename} → {conference}")
        print("================")
        
        faculty_df = process_single_file(csv_file, conference, author_univ_map)
        
        if not faculty_df.empty:
            accumulated_rankings = update_university_rankings(faculty_df, accumulated_rankings)
            processed_count = len(faculty_df)
            total_processed += processed_count
            
            results.append({
                'file': filename,
                'conference': conference,
                'processed': processed_count,
                'status': 'succces'
            })
            print(f"Added {processed_count} faculty to rankings")
        else:
            results.append({
                'file': filename,
                'conference': conference,
                'processed': 0,
                'status': 'no_match'
            })
            print(f"No matches found")
    
    if accumulated_rankings is not None and not accumulated_rankings.empty:
        print(f"\nTrying...")
   
        EXPECTED_COLUMNS = [
            'Index', 'University', 'Artificial Intelligence & Machine Learning',
            'Data Science & Data Mining', 'Computer Vision & Image Processing', 
            'Natural Language Processing', 'Systems & Networking', 'Databases',
            'Security & Privacy', 'Human Computer Interaction', 'Theoretical Computer Science',
            'Software Engineering', 'Computer Graphics & Virtual Reality', 
            'Quantum Computing', 'Interdisciplinary Fields', 'Continent'
        ]
        
        accumulated_rankings = accumulated_rankings.reindex(columns=EXPECTED_COLUMNS)
        
        accumulated_rankings.to_csv(output_file, index=False)
        
        print(f"\n====== Done: ======\n")
        print(f"Total faculty processed: {total_processed:,}")
        print(f"Total universities: {len(accumulated_rankings):,}")
        print(f"Output saved to: {output_file}")
        # print(f"Format: Standard university rankings CSV")
        
        # print("Columns:", ", ".join(accumulated_rankings.columns[:5]), "...", accumulated_rankings.columns[-1])
        
        if len(accumulated_rankings) > 0:
            first_row = accumulated_rankings.iloc[0]
            print(f"Sample row: {first_row['Index']}, {first_row['University']}, {first_row.iloc[2]:.1f}...")
        
        print(f"\nResults breakdown:")
        for result in results:
            statuse = {'success': 'yes', 'no_match': 'no'}.get(result['status'], '?')
            print(f"  {statuse} {result['file']:<35} | {result['conference']:<8} | {result['processed']:>5,} faculty")
    
    else:
        print("\nNo data processed successfully.")

def single_file_mode(faculty_folder):
    faculty_folder = Path(faculty_folder)
    csv_files = list(faculty_folder.glob("*.csv"))
    
    print(f"\n SINGLE FILE MODE")
    print("Available files:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file.name}")
    
    choice = input(f"\nSelect file number (1-{len(csv_files)}): ").strip()
    
    if not choice.isdigit() or not 1 <= int(choice) <= len(csv_files):
        print("Invalid selection")
        return
    
    selected_file = csv_files[int(choice) - 1]
    
    suggested = smart_extract_conference(selected_file.name)
    if suggested:
        conference = input(f"Conference (suggested: {suggested}): ").strip() or suggested
    else:
        conference = input("Enter conference name: ").strip()
    
    if conference.upper() not in CONFERENCE_CATEGORIES:
        print(f"Unknown conference: {conference}")
        return
    
    print(f"\nProcessing {selected_file.name} as {conference.upper()}...")
    
    try:
        csrankings_df = load_csrankings_data()
        author_univ_map = create_author_university_mapping(csrankings_df)
        
        faculty_df = process_single_file(selected_file, conference.upper(), author_univ_map)
        
        if not faculty_df.empty:
            rankings = update_university_rankings(faculty_df)
            
            EXPECTED_COLUMNS = [
                'Index', 'University', 'Artificial Intelligence & Machine Learning',
                'Data Science & Data Mining', 'Computer Vision & Image Processing', 
                'Natural Language Processing', 'Systems & Networking', 'Databases',
                'Security & Privacy', 'Human Computer Interaction', 'Theoretical Computer Science',
                'Software Engineering', 'Computer Graphics & Virtual Reality', 
                'Quantum Computing', 'Interdisciplinary Fields', 'Continent'
            ]
            rankings = rankings.reindex(columns=EXPECTED_COLUMNS)
            
            output_file = f"{selected_file.stem}_{conference.lower()}_rankings.csv"
            rankings.to_csv(output_file, index=False)
            
            print(f"Processed {len(faculty_df)} faculty")
            print(f"Total universities: {len(rankings)}")
            print(f"Output saved to: {output_file}")
        else:
            print("No matches found")
            
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    faculty_folder = Path("../faculty/Scoring")
    output_file = Path("3cv_f.csv")
    
    main_interactive_processor(faculty_folder, output_file)