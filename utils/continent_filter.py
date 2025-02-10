import pandas as pd

# Load the university rankings CSV
university_df = pd.read_csv("universities_ranked.csv")

# Load the institution mapping CSV
country_df = pd.read_csv("country-info.csv")

# Rename 'region' to 'Continent' for clarity
country_df = country_df.rename(columns={"region": "Continent"})

# Merge datasets on the 'University' and 'institution' columns
updated_df = university_df.merge(country_df[["institution", "Continent"]], 
                                 left_on="University", 
                                 right_on="institution", 
                                 how="left")

# Fill in empty 'Continent' values with 'North America'
updated_df["Continent"].fillna("North America", inplace=True)

# Drop the extra 'institution' column after merging
updated_df.drop(columns=["institution"], inplace=True)

# Save updated CSV
updated_df.to_csv("updated_university_rankings.csv", index=False)

print("Updated CSV saved as 'updated_university_rankings.csv'.")