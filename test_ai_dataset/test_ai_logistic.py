
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

    # Dataset loading
    df = pd.read_csv(dataset_file)

    x = df["text_combined"].fillna("")
    y = df["label"].astype(int).values

    # Load the already trained TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_file)

    # Text transformation
    x_tfidf = vectorizer.transform(x)

    # Load the already trained model
    model = joblib.load(model_file)

    # Prediction
    predictions = model.predict(x_tfidf)

    # Overall metrics (positive class = 1, phishing)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y, predictions, pos_label=1, zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"Tested emails: {len(df)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion matrix details:")
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

    # Results to be collected in the summary
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

# Dataset 1
results.append(test_model(
    "../models/phishing_model_logistic_dataset1.pkl",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email.csv",
    "Logistic Dataset 1 -> AI Italian"
))

results.append(test_model(
    "../models/phishing_model_logistic_dataset1.pkl",
    "../vectorizers/tfidf_vectorizer_dataset1.pkl",
    "../datasets/ai_email_en.csv",
    "Logistic Dataset 1 -> AI English"
))

# Dataset 2
results.append(test_model(
    "../models/phishing_model_logistic_dataset2.pkl",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email.csv",
    "Logistic Dataset 2 -> AI Italian"
))

results.append(test_model(
    "../models/phishing_model_logistic_dataset2.pkl",
    "../vectorizers/tfidf_vectorizer_dataset2.pkl",
    "../datasets/ai_email_en.csv",
    "Logistic Dataset 2 -> AI English"
))


# =========================
# FINAL SUMMARY
# =========================

results_df = pd.DataFrame(results)

# Print metrics as percentages
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
print("RESULT SUMMARY (%)")
print("=" * 70)
print(results_display.to_string(index=False))

# Save numerical results in CSV format
results_df.to_csv(
    "../datasets/results_logistic_ai.csv",
    index=False
)

print("\nResults summary saved to in: ../datasets/results_logistic_ai.csv")
