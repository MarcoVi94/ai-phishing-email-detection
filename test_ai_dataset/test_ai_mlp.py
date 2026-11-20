
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# ARCHITETTURA MLP
# =========================

class PhishingClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================
# FUNZIONE DI TEST
# =========================

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

    # Conversione in tensore
    x_tensor = torch.tensor(
        x_tfidf.toarray(),
        dtype=torch.float32
    )

    # Caricamento modello già addestrato
    model = PhishingClassifier(x_tensor.shape[1])

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )

    model.eval()

    # Predizione
    with torch.no_grad():
        outputs = model(x_tensor)
        predictions = (torch.sigmoid(outputs) >= 0.5).int()

    predictions = predictions.numpy().flatten()

    # Metriche complessive
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y, predictions, pos_label=1, zero_division=0
    )

    # Matrice di confusione: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Stampa metriche
    print(f"Email testate: {len(df)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nDettaglio matrice di confusione:")
    print(f"TN (legittime classificate come legittime): {tn}")
    print(f"FP (legittime classificate come phishing): {fp}")
    print(f"FN (phishing classificato come legittimo): {fn}")
    print(f"TP (phishing classificato come phishing): {tp}")

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Legittima", "Phishing"],
        digits=4,
        zero_division=0
    ))

    # Risultati per il riepilogo finale
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

# DATASET 1

results.append(test_model(
    "../models/phishing_model_mlp_dataset1.pth",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email.csv",
    "MLP Dataset 1 -> AI Italiano"
))

results.append(test_model(
    "../models/phishing_model_mlp_dataset1.pth",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email_en.csv",
    "MLP Dataset 1 -> AI Inglese"
))


# DATASET 2

results.append(test_model(
    "../models/phishing_model_mlp_dataset2.pth",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email.csv",
    "MLP Dataset 2 -> AI Italiano"
))

results.append(test_model(
    "../models/phishing_model_mlp_dataset2.pth",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email_en.csv",
    "MLP Dataset 2 -> AI Inglese"
))


# =========================
# RIEPILOGO FINALE
# =========================

results_df = pd.DataFrame(results)

# Conversione delle metriche in percentuale
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

# Salvataggio del riepilogo
results_df.to_csv(
    "../datasets/risultati_mlp_ai.csv",
    index=False
)

print("\nRiepilogo salvato in:")
print("../datasets/risultati_mlp_ai.csv")