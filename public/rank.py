import pandas as pd

df = pd.read_csv("cv_update.csv")

df_sorted = df.sort_values(by="Computer Vision & Image Processing", ascending=False)

df_sorted.to_csv("ranked_cv.csv", index=False)

print(df_sorted.head())
