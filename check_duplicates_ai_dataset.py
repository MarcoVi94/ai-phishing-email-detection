import pandas as pd

df = pd.read_csv("datasets/ai_email.csv")

print("Total number of emails:", len(df))

duplicates = df[df.duplicated(subset="text_combined", keep=False)]

print("Number of emails involved in duplicates:", len(duplicates))
print("Number of duplicates texts:", df["text_combined"].duplicated().sum())

if len(duplicates) > 0:
    print("\nDuplicate emails:")
    print(duplicates[["category", "text_combined"]].to_string(index=False))
else:
    print("\nNo duplicates found.")

df_without_duplicates = df.drop_duplicates(subset="text_combined")

print("\nNumber of emails after removing duplicates:", len(df_without duplicates))
