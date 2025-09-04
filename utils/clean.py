import pandas as pd

df = pd.read_csv("faculty_contributions.csv")

df_clean = df[df["University"].str.strip() != "Not Found"]

df_clean.to_csv("faculty_score.csv", index=False)

print(df_clean.head())