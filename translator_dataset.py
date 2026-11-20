import pandas as pd
import argostranslate.translate
import os

INPUT_FILE = "datasets/ai_email.csv"
OUTPUT_FILE = "datasets/ai_email_en.csv"

df = pd.read_csv(INPUT_FILE)

# Se esiste già un file tradotto, riprende da quello
if os.path.exists(OUTPUT_FILE):
    df_en = pd.read_csv(OUTPUT_FILE)

    translated_count = df_en["text_combined"].notna().sum()

    print(f"Email già tradotte: {translated_count}")

else:
    df_en = df.copy()
    df_en["text_combined"] = None
    translated_count = 0

total = len(df)

print(f"Totale email: {total}")

for i in range(translated_count, total):

    text = str(df.loc[i, "text_combined"])

    try:
        translated = argostranslate.translate.translate(
            text,
            "it",
            "en"
        )

        df_en.loc[i, "text_combined"] = translated

        # Salva dopo ogni email
        df_en.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

        print(f"Tradotta email {i + 1}/{total}")

    except Exception as e:

        print(f"Errore alla email {i + 1}: {e}")
        print("Le traduzioni già completate sono state salvate.")
        break

print("\nTraduzione completata.")
print(f"File salvato: {OUTPUT_FILE}")