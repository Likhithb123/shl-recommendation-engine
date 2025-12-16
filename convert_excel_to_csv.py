import pandas as pd
import os

os.makedirs("datasets", exist_ok=True)

df = pd.read_excel("Gen_AI Dataset.xlsx")
df.to_csv("datasets/shl_labeled.csv", index=False)

print("Saved datasets/shl_labeled.csv")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
