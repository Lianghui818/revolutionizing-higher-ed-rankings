import pandas as pd
import glob

def extract_authors_universities():
    #Combine all paper CSVs that we currently have, we can add more later. 
    papers_files = glob.glob("./CVPR_*.csv") + glob.glob("./ICML_*.csv") + glob.glob("./NeurIPS_*.csv")
    papers_list = [pd.read_csv(file) for file in papers_files]
    papers = pd.concat(papers_list, ignore_index=True)

    #Combine all csrankings-(a-z).csv files from CSrankings git repo.
    csrankings_files = glob.glob("../CSrankings/csrankings-*.csv")
    csrankings_list = [pd.read_csv(file) for file in csrankings_files]
    csrankings = pd.concat(csrankings_list, ignore_index=True)

    #Create a dictionary mapping author names to their universities.
    author_university_map = dict(zip(csrankings['name'], csrankings['affiliation']))

    #Initialize a set to store authors and their universities, exclude authors that appear multiple times acorss csvs. 
    author_university_set = set()

    #Iterate through the combined papers data.
    for index, row in papers.iterrows():

        
        if pd.isna(row['Authors']): #Rows with missing authors.
            continue

        authors = row['Authors'].split(",") 
        for author in authors:
            author = author.strip()  
            university = author_university_map.get(author)  
            if university:  #Add authors with found universities, ingnore an author if there's no matching data in csrankings-(a-z).csv. 
                author_university_set.add((author, university))

    #Convert the results into a DataFrame
    result_df = pd.DataFrame(list(author_university_set), columns=["Author", "University"])

    #Print the result to output csv file
    result_df.to_csv("author_universities_output.csv", index=False)

    return result_df

author_university_df = extract_authors_universities()
