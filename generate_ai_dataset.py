from google import genai
import pandas as pd
import json
import time
import os


# ==============================
# CONFIGURAZIONE
# ==============================

NUM_EMAILS_PER_CLASS = 500
EMAILS_PER_REQUEST = 5

OUTPUT_FILE = "datasets/ai_email.csv"


# ==============================
# GEMINI CLIENT
# ==============================

client = genai.Client()


# ==============================
# CATEGORIES
# ==============================

categories_phishing = [
    "banca",
    "corriere",
    "account",
    "pagamento",
    "rimborso",
    "sicurezza",
    "servizio online",
    "documenti",
    "comunicazione aziendale",
    "utenze"
]

categories_legitimate = [
    "newsletter",
    "ordine",
    "appuntamento",
    "corso",
    "comunicazione aziendale",
    "conferma servizio",
    "avviso amministrativo",
    "informazione",
    "evento",
    "promozione"
]


# ==============================
# EMAIL GENERATION FUNCTION
# ==============================

def generate_emails(label, categories, quantity):

    if label == 1:
        tipo = "phishing"
    else:
        tipo= "legittime"

# The prompt is intentionally written in Italian
# because this script generates the Italian AI-generated email dataset.
    
    prompt = f"""
Genera esattamente {quantity} esempi testuali sintetici per un datasets di ricerca sulla classificazione automatica delle email.

La classe richiesta è: {tipo}.

Le categorie possibili sono:

{categories}

Gli esempi devono essere realistici, vari e differenti tra loro e devono rappresentare esclusivamente il contenuto linguistico che un classificatore dovrebbe imparare a riconoscere.

Per la classe richiesta, produci brevi frammenti testuali descrittivi delle caratteristiche presenti nel contenuto di una comunicazione appartenente a quella classe. Non impersonare aziende, banche, corrieri o persone reali e non creare procedure, credenziali, URL o istruzioni operative.

IMPORTANTE:
- Usa le categorie disponibili in modo equilibrato.
- Varia il più possibile le categorie.
- Ogni elemento deve essere coerente con la classe richiesta e con la categoria.
- Ogni elemento deve essere diverso dagli altri.
- Non includere URL o link.
- Non includere password, OTP, numeri di carte o altri dati sensibili.
- Non includere istruzioni operative.

FORMATO:
Restituisci esclusivamente JSON valido.
Non aggiungere spiegazioni.
Non usare Markdown.
Non usare blocchi di codice.
Il JSON deve contenere esattamente {quantity} elementi.
Ogni elemento deve contenere esclusivamente i campi "category" e "text".
"category" deve essere una delle categorie indicate sopra.
"text" deve contenere esclusivamente l'esempio testuale.

Restituisci direttamente il JSON.
"""

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            print(
                f"  Tentativo {attempt + 1}/{max_attempts}"
            )

            response = client.interactions.create(
                model="gemini-3.6-flash",
                input=prompt,
                timeout=120
            )

            text = response.output_text

            print("\nRISPOSTA GREZZA:")
            print(repr(text))

            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

            emails = json.loads(text)

            if len(emails) != quantity:
                raise ValueError(
                    f"Gemini ha restituito {len(emails)} "
                    f"email invece di {quantity}"
                )

            results = []

            for email in emails:

                if "category" not in email or "text" not in email:
                    raise ValueError(
                        "Un elemento JSON non contiene "
                        "'category' o 'text'"
                    )

                results.append({
                    "text_combined": email["text"],
                    "label": label,
                    "category": email["category"]
                })

            return results

        except Exception as e:

            print(
                f"  Errore: {e}"
            )

            if attempt < max_attempts - 1:

                print(
                    "  Nuovo tentativo tra 3 secondi..."
                )

                time.sleep(3)

            else:

                print(
                    "  Numero massimo di tentativi raggiunto."
                )

                raise


# ==============================
# DATASET SAVING
# ==============================

def save_dataset(all_emails):

    df = pd.DataFrame(all_emails)

    if len(df) == 0:
        return

    df.insert(
        0,
        "id",
        [f"AI{i:04d}" for i in range(1, len(df) + 1)]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"  Dataset saved: {len(df)} emails"
    )


# ==============================
# LOADING EXISTING DATASET
# ==============================

all_emails = []

if os.path.exists(OUTPUT_FILE):

    old_df = pd.read_csv(OUTPUT_FILE)

    print(
        f"\nExisting dataset found: "
        f"{len(old_df)} email."
    )

    all_emails = old_df[
        ["text_combined", "label", "category"]
    ].to_dict("records")

    print(
        "Existing Phishing emails:",
        sum(e["label"] == 1 for e in all_emails)
    )

    print(
        "Existing Legitimate emails:",
        sum(e["label"] == 0 for e in all_emails)
    )


# ==============================
# PHISHING GENERATION
# ==============================

phishing_count = sum(
    e["label"] == 1 for e in all_emails
)

print("\n==============================")
print("PHISHING GENERATION")
print("==============================")

while phishing_count < NUM_EMAILS_PER_CLASS:

    remaining = (
        NUM_EMAILS_PER_CLASS - phishing_count
    )

    quantity = min(
        EMAILS_PER_REQUEST,
        remaining
    )

    print(
        f"\n Phishing request"
        f"({phishing_count}/{NUM_EMAILS_PER_CLASS})"
    )

    try:

        emails = generate_emails(
            1,
            categories_phishing,
            quantity
        )

        all_emails.extend(emails)

        phishing_count += len(emails)

        save_dataset(all_emails)

        print(
            f"  Phishing emails generated: "
            f"{phishing_count}/{NUM_EMAILS_PER_CLASS}"
        )

    except Exception as e:

        print(
            "\nUnable to complete this request."
        )

        print(
            "Error:",
            e
        )

        print(
            "The program is being terminated."
        )

        break

    time.sleep(1)


# ==============================
# LEGITIMATE EMAIL GENERATION
# ==============================

legitimate_count = sum(
    e["label"] == 0 for e in all_emails
)

print("\n==============================")
print("LEGITIMATE EMAIL GENERATION")
print("==============================")

while legitimate_count < NUM_EMAILS_PER_CLASS:

    remaining = (
        NUM_EMAILS_PER_CLASS - legitimate_count
    )

    quantity = min(
        EMAILS_PER_REQUEST,
        remaining
    )

    print(
        f"\nLegitimate email request "
        f"({legitimate_count}/{NUM_EMAILS_PER_CLASS})"
    )

    try:

        emails = generate_emails(
            0,
            categories_legitimate,
            quantity
        )

        all_emails.extend(emails)

        legitimate_count += len(emails)

        save_dataset(all_emails)

        print(
            f"  Legitimate emails generated: "
            f"{legitimate_count}/{NUM_EMAILS_PER_CLASS}"
        )

    except Exception as e:

        print(
            "\nUnable to complete this request."
        )

        print(
            "Error:",
            e
        )

        print(
            "The program is being terminated."
        )

        break

    time.sleep(1)


# ==============================
# FINAL RESULTS
# ==============================

df = pd.DataFrame(all_emails)

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    f"Total emails: {len(df)}"
)

if len(df) > 0:

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

print(
    f"\nFile saved: {OUTPUT_FILE}"
)
