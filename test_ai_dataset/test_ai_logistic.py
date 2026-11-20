
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def test_model(model_file, vectorizer_file, dataset_file, dataset_name):

    print("\n" + "=" * 70)
    print(f"TEST: {dataset_name}")
    print("=" * 70)

    # Caricamento dataset
    df = pd.read_csv(dataset_file)

    x = df["text_combined"].fillna("")
    y = df["label"].astype(int).values

    # Caricamento TF-IDF già addestrato
    vectorizer = joblib.load(vectorizer_file)

    # Trasformazione del testo
    x_tfidf = vectorizer.transform(x)

    # Caricamento modello già addestrato
    model = joblib.load(model_file)

    # Predizione
    predictions = model.predict(x_tfidf)

    # Metriche complessive (classe positiva = 1, phishing)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y, predictions, pos_label=1, zero_division=0)

    # Matrice di confusione: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"Email testate: {len(df)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nDettaglio matrice di confusione:")
    print(f"TN (email legittime classificate come legittime): {tn}")
    print(f"FP (email legittime classificate come phishing): {fp}")
    print(f"FN (email phishing classificate come legittime): {fn}")
    print(f"TP (email phishing classificate come phishing): {tp}")

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Legittima", "Phishing"],
        digits=4,
        zero_division=0
    ))

    # Risultati da raccogliere nel riepilogo
    return {
        "Esperimento": dataset_name,
        "Email testate": len(df),
        "Accuracy": accuracy,
        "Precision phishing": precision,
        "Recall phishing": recall,
        "F1-score phishing": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# =========================
# ESECUZIONE DEI TEST
# =========================

results = []

# Dataset 1
results.append(test_model(
    "../models/phishing_model_logistic_dataset1.pkl",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email.csv",
    "Logistic Dataset 1 -> AI Italiano"
))

results.append(test_model(
    "../models/phishing_model_logistic_dataset1.pkl",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email_en.csv",
    "Logistic Dataset 1 -> AI Inglese"
))

# Dataset 2
results.append(test_model(
    "../models/phishing_model_logistic_dataset2.pkl",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email.csv",
    "Logistic Dataset 2 -> AI Italiano"
))

results.append(test_model(
    "../models/phishing_model_logistic_dataset2.pkl",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email_en.csv",
    "Logistic Dataset 2 -> AI Inglese"
))


# =========================
# RIEPILOGO FINALE
# =========================

results_df = pd.DataFrame(results)

# Stampa delle metriche in percentuale
metric_columns = [
    "Accuracy",
    "Precision phishing",
    "Recall phishing",
    "F1-score phishing"
]

results_display = results_df.copy()
results_display[metric_columns] = (
    results_display[metric_columns] * 100
).round(2)

print("\n" + "=" * 70)
print("RIEPILOGO DEI RISULTATI (%)")
print("=" * 70)
print(results_display.to_string(index=False))

# Salvataggio dei risultati numerici in formato CSV
results_df.to_csv(
    "../datasets/risultati_logistic_ai.csv",
    index=False
)

print("\nRiepilogo salvato in: ../datasets/risultati_logistic_ai.csv")