import pandas as pd

df = pd.read_csv("datasets/ai_email.csv")

print("Numero totale di email:", len(df))

duplicati = df[df.duplicated(subset="text_combined", keep=False)]

print("Numero di email coinvolte in duplicati:", len(duplicati))
print("Numero di testi duplicati:", df["text_combined"].duplicated().sum())

if len(duplicati) > 0:
    print("\nEmail duplicate:")
    print(duplicati[["category", "text_combined"]].to_string(index=False))
else:
    print("\nNon sono presenti duplicati.")

df_senza_duplicati = df.drop_duplicates(subset="text_combined")

print("\nEmail dopo la rimozione dei duplicati:", len(df_senza_duplicati))