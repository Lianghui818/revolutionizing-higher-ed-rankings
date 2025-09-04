import pandas as pd

df = pd.read_csv("3cv_f.csv")

df_sorted = df.sort_values(by="Computer Vision & Image Processing", ascending=False)

df_sorted.to_csv("ranked_3cv_f.csv", index=False)

print(df_sorted.head())
