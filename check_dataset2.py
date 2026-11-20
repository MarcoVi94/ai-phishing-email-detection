import pandas as pd

df = pd.read_csv("datasets/AVN_Corpus.csv")

print("Dimensioni:", df.shape)

print("\nColonne:")
print(df.columns.tolist())

print("\nDistribuzione label:")
print(df["label"].value_counts())

print("\nDuplicati:")
print(df.duplicated(subset=["body"]).sum())