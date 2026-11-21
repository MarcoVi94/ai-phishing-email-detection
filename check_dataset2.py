import pandas as pd

df = pd.read_csv("datasets/AVN_Corpus.csv")

print("Dataset dimensions:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nDuplicates:")
print(df.duplicated(subset=["body"]).sum())
