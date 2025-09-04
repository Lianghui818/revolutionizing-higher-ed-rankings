import pandas as pd
from pathlib import Path
import requests
from io import StringIO
import string
import json
from collections import defaultdict

def load_csrankings_mapping():
    print("[INFO] Loading CSrankings author-university mapping...")
    
    all_authors = []
    letters = string.ascii_lowercase
    
    for letter in letters:
        url = f"https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/csrankings-{letter}.csv"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                all_authors.append(df)
        except:
            continue
    
    combined_df = pd.concat(all_authors, ignore_index=True)
    
    if 'Name' in combined_df.columns:
        combined_df['name'] = combined_df['Name']
    if 'Affiliation' in combined_df.columns:
        combined_df['affiliation'] = combined_df['Affiliation']
    
    clean_df = combined_df.dropna(subset=['name', 'affiliation'])
    author_univ_map = clean_df.set_index('name')['affiliation'].to_dict()
    
    print(f"[INFO] Loaded {len(author_univ_map)} author-university mappings")
    return author_univ_map

def extract_conference_name(filename):

    base_name = Path(filename).stem.upper()
    
    conferences = {
        'CVPR': 'Computer Vision & Image Processing',
        'ICCV': 'Computer Vision & Image Processing', 
        'ECCV': 'Computer Vision & Image Processing',
        'AAAI': 'Artificial Intelligence & Machine Learning',
        'IJCAI': 'Artificial Intelligence & Machine Learning',
        'NeurIPS': 'Artificial Intelligence & Machine Learning',
        'ICML': 'Artificial Intelligence & Machine Learning',
        'ACL': 'Natural Language Processing',
        'EMNLP': 'Natural Language Processing'
    }
    
    for conf_name, category in conferences.items():
        if conf_name in base_name:
            return conf_name, category
    
    return "Unknown", "Unknown"

def generate_faculty_contribution_report(faculty_folder, output_file="faculty_contributions.csv"):
    
    print("Faculty report:\n\n")

    author_univ_map = load_csrankings_mapping()
    
    faculty_folder = Path(faculty_folder)
    csv_files = list(faculty_folder.glob("*.csv"))
    
    all_contributions = []
    
    print(f"Processing {len(csv_files)} files...")
    
    for csv_file in csv_files:
        print(f"   Processing: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            
            name_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['full name', 'name', 'author', 'faculty']):
                    name_col = col
                    break
            
            if not name_col:
                print(f"   ⚠️  No name column found in {csv_file.name}")
                continue
            
            conference, category = extract_conference_name(csv_file.name)
            
            for _, row in df.iterrows():
                name = str(row[name_col]).strip()
                if not name or name == 'nan':
                    continue
                
                university = author_univ_map.get(name, "Not Found")
                
                score = float(row.get("Normalized Score", 1))
                
                contribution = {
                    'Faculty Name': name,
                    'University': university,
                    'Conference': conference,
                    'Category': category,
                    'Score': score,
                    'Source File': csv_file.name,
                    'Matched': 'Yes' if university != "Not Found" else 'No'
                }
                
                all_contributions.append(contribution)
        
        except Exception as e:
            print(f"!!!!!!!!!!Error processing {csv_file.name}: {e}")
    
    contributions_df = pd.DataFrame(all_contributions)
    
    if contributions_df.empty:
        print("No contributions found!")
        return None
    
    contributions_df.to_csv(output_file, index=False)
    
    print(f"\nCntribution summary:")
    print(f"   Total faculty entries: {len(contributions_df):,}")
    print(f"   Successfully matched: {len(contributions_df[contributions_df['Matched'] == 'Yes']):,}")
    print(f"   Match rate: {len(contributions_df[contributions_df['Matched'] == 'Yes']) / len(contributions_df) * 100:.1f}%")
    print(f"   Unique faculty: {contributions_df['Faculty Name'].nunique():,}")
    print(f"   Unique universities: {contributions_df[contributions_df['Matched'] == 'Yes']['University'].nunique():,}")
    print(f"   Report saved to: {output_file}")
    
    return contributions_df

def analyze_university_details(contributions_df, university_name):
    
    print(f"\nDetaled information for: {university_name}\n\n")

    university_data = contributions_df[
        contributions_df['University'].str.contains(university_name, case=False, na=False)
    ]
    
    if university_data.empty:
        print(f"No data found for university containing '{university_name}'")
        print("\nAvailable universities containing similar terms:")
        
        for word in university_name.split():
            if len(word) > 3:
                similar = contributions_df[
                    contributions_df['University'].str.contains(word, case=False, na=False)
                ]['University'].unique()
                
                if len(similar) > 0:
                    print(f"   Universities containing '{word}':")
                    for univ in similar[:10]:
                        count = len(contributions_df[contributions_df['University'] == univ])
                        print(f"     - {univ} ({count} faculty)")
        return
    
    exact_universities = university_data['University'].unique()
    
    print(f"Found {len(exact_universities)} university name(s):")
    for univ in exact_universities:
        count = len(university_data[university_data['University'] == univ])
        print(f"   - {univ} ({count} faculty entries)")
    
    print(f"\nOVERALL STATISTICS:")
    total_faculty = len(university_data)
    unique_faculty = university_data['Faculty Name'].nunique()
    total_score = university_data['Score'].sum()
    
    print(f"   Total faculty entries: {total_faculty}")
    print(f"   Unique faculty: {unique_faculty}")
    print(f"   Total score: {total_score:.2f}")
    print(f"   Average score per entry: {total_score/total_faculty:.2f}")
    
    print(f"\nBREAKDOWN BY CONFERENCE:")
    by_conference = university_data.groupby('Conference').agg({
        'Faculty Name': 'count',
        'Score': 'sum'
    }).sort_values('Score', ascending=False)
    
    for conf, data in by_conference.iterrows():
        print(f"   {conf}: {data['Faculty Name']} faculty, {data['Score']:.2f} points")
    
    print(f"\nBREAKDOWN BY CATEGORY:")
    by_category = university_data.groupby('Category').agg({
        'Faculty Name': 'count', 
        'Score': 'sum'
    }).sort_values('Score', ascending=False)
    
    for cat, data in by_category.iterrows():
        print(f"   {cat}: {data['Faculty Name']} faculty, {data['Score']:.2f} points")

    print(f"\nTOP CONTRIBUTING FACULTY:")
    top_faculty = university_data.groupby('Faculty Name').agg({
        'Score': 'sum',
        'Conference': 'count',
        'Category': lambda x: ', '.join(x.unique())
    }).sort_values('Score', ascending=False)
    
    for i, (name, data) in enumerate(top_faculty.head(15).iterrows(), 1):
        print(f"{i:2d}. {name}")
        print(f"    Total Score: {data['Score']:.2f}")
        print(f"    Entries: {data['Conference']}")
        print(f"    Categories: {data['Category']}")
        print()
    
    print(f"\nCHECKING FOR DUPLICATE CONTRIBUTIONS:")
    duplicates = university_data[university_data.duplicated(['Faculty Name', 'Conference'], keep=False)]
    if not duplicates.empty:
        print(f"   Found {len(duplicates)} duplicate entries:")
        for name in duplicates['Faculty Name'].unique()[:5]:
            dups = duplicates[duplicates['Faculty Name'] == name]
            print(f"   - {name}: appears {len(dups)} times in same conference")
    else:
        print(f"   No duplicate entries found")
    
    return university_data

def compare_universities(contributions_df, university_names):
    
    print(f"\nUNIVERSITY COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for univ_name in university_names:
        univ_data = contributions_df[
            contributions_df['University'].str.contains(univ_name, case=False, na=False)
        ]
        
        if not univ_data.empty:
            exact_names = univ_data['University'].unique()
            
            total_faculty = len(univ_data)
            unique_faculty = univ_data['Faculty Name'].nunique()
            total_score = univ_data['Score'].sum()
            
            by_category = univ_data.groupby('Category')['Score'].sum()
            
            comparison_data.append({
                'Search Term': univ_name,
                'Exact Names': exact_names,
                'Total Faculty Entries': total_faculty,
                'Unique Faculty': unique_faculty,
                'Total Score': total_score,
                'Score per Faculty': total_score / unique_faculty if unique_faculty > 0 else 0,
                'Category Breakdown': by_category.to_dict()
            })
    
    print(f"COMPARISON RESULTS:")
    print()
    
    for data in sorted(comparison_data, key=lambda x: x['Total Score'], reverse=True):
        print(f"  {data['Search Term'].upper()}")
        if len(data['Exact Names']) > 1:
            print(f"Exact matches: {', '.join(data['Exact Names'])}")
        else:
            print(f"Exact name: {data['Exact Names'][0]}")
        
        print(f"   Unique faculty: {data['Unique Faculty']}")
        print(f"   Total score: {data['Total Score']:.2f}")
        print(f"   Score per faculty: {data['Score per Faculty']:.2f}")
        
        print(f"Category scores:")
        for cat, score in sorted(data['Category Breakdown'].items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                print(f"      - {cat}: {score:.2f}")
        print()

def main_faculty_analysis():
    
    print("FACULTY CONTRIBUTION DETAILED TRACKER")
    print("="*80)
    
    # path
    faculty_folder = "../faculty/Scoring"
    
    print("Select analysis option:")
    print("1. Generate complete faculty contribution report")
    print("2. Analyze specific university")
    print("3. Compare multiple universities")
    print("4. Load existing report and analyze")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    contributions_df = None
    
    if choice == '1':
        contributions_df = generate_faculty_contribution_report(faculty_folder)
        
    elif choice == '4':
        report_file = input("Enter report file name (default: faculty_contributions.csv): ").strip()
        if not report_file:
            report_file = "faculty_contributions.csv"
        
        try:
            contributions_df = pd.read_csv(report_file)
            print(f" Loaded existing report: {report_file}")
        except:
            print(f" Cannot load {report_file}")
            return
    
    else:
        if Path("faculty_contributions.csv").exists():
            print("📄 Found existing faculty_contributions.csv")
            use_existing = input("Use existing report? (y/n): ").strip().lower()
            if use_existing in ['y', 'yes']:
                contributions_df = pd.read_csv("faculty_contributions.csv")
            else:
                contributions_df = generate_faculty_contribution_report(faculty_folder)
        else:
            print("📄 Generating new faculty contribution report...")
            contributions_df = generate_faculty_contribution_report(faculty_folder)
    
    if contributions_df is None:
        return
    

    if choice == '2' or choice == '4':
        university = input("Enter university name to analyze: ").strip()
        if university:
            analyze_university_details(contributions_df, university)
    
    elif choice == '3':
        universities_input = input("Enter university names (comma-separated): ").strip()
        if universities_input:
            universities = [u.strip() for u in universities_input.split(',')]
            compare_universities(contributions_df, universities)

if __name__ == "__main__":
    main_faculty_analysis()