
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
# MLP ARCHITECTURE
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
# TEST FUNCTION
# =========================

def test_model(model_file, vectorizer_file, dataset_file, dataset_name):

    print("\n" + "=" * 70)
    print(f"TEST: {dataset_name}")
    print("=" * 70)

    # Dataset loading
    df = pd.read_csv(dataset_file)

    x = df["text_combined"].fillna("")
    y = df["label"].astype(int).values

    # Load the already trained TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_file)

    # Text transformation
    x_tfidf = vectorizer.transform(x)

    # Tensor conversion
    x_tensor = torch.tensor(
        x_tfidf.toarray(),
        dtype=torch.float32
    )

    # Load the already trained model
    model = PhishingClassifier(x_tensor.shape[1])

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )

    model.eval()

    # Prediction
    with torch.no_grad():
        outputs = model(x_tensor)
        predictions = (torch.sigmoid(outputs) >= 0.5).int()

    predictions = predictions.numpy().flatten()

    # Overall metrics
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

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Print metrics
    print(f"Tested emails: {len(df)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Details:")
    print(f"TN (legitimate emails classified as legitimate): {tn}")
    print(f"FP (legitimate emails classified as phishing): {fp}")
    print(f"FN (phishing emails classified as legitimate): {fn}")
    print(f"TP (phishing emails classified as phishing): {tp}")

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        digits=4,
        zero_division=0
    ))

    # Results for the final summary
    return {
        "Experiment": dataset_name,
        "Tested emails": len(df),
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
# RUNNING THE TESTS
# =========================

results = []

# DATASET 1

results.append(test_model(
    "../models/phishing_model_mlp_dataset1.pth",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email.csv",
    "MLP Dataset 1 -> AI Italian"
))

results.append(test_model(
    "../models/phishing_model_mlp_dataset1.pth",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email_en.csv",
    "MLP Dataset 1 -> AI English"
))


# DATASET 2

results.append(test_model(
    "../models/phishing_model_mlp_dataset2.pth",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email.csv",
    "MLP Dataset 2 -> AI Italian"
))

results.append(test_model(
    "../models/phishing_model_mlp_dataset2.pth",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email_en.csv",
    "MLP Dataset 2 -> AI English"
))


# =========================
# FINAL SUMMARY
# =========================

results_df = pd.DataFrame(results)

# Convert metrics to percentages
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
print("RESULTS SUMMARY (%)")
print("=" * 70)
print(results_display.to_string(index=False))

# Save the summary
results_df.to_csv(
    "../datasets/results_mlp_ai.csv",
    index=False
)

print("\nSummary saved to:")
print("../datasets/results_mlp_ai.csv")
