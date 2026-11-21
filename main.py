import torch
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from mlp import run_mlp
from logistic_regression import run_logistic_regression


# =========================
# DATASET LOADING
# =========================

DATASET = 2

if DATASET == 1:
    dataset_path = "datasets/phishing_email.csv"
    suffix = "dataset1"

elif DATASET == 2:
    dataset_path = "datasets/AVN_Corpus.csv"
    suffix = "dataset2"


df = pd.read_csv(dataset_path)

# =========================
# PREPROCESSING
# =========================

if DATASET == 1:

    # Dataset 1:
    # The email text is already contained
    # in the text_combined column
    df["text_combined"] = df["text_combined"].fillna("")


elif DATASET == 2:

    # Dataset 2:
    # Combine subject and body to create
    # the complete email text
    df["text_combined"] = (
        df["subject"].fillna("") + " " +
        df["body"].fillna("")
    )

# Remove duplicates
df = df.drop_duplicates(subset="text_combined").copy()

print("Dimensions after removing duplicates:", df.shape)


# =========================
# INPUT AND LABELS
# =========================

# X contains the emails
# y contains the classifications
x = df["text_combined"]
y = df["label"]


# =========================
# TRAIN / TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    # 20% of the data will be used for testing
    test_size=0.20,

    # The split will always be the same
    random_state=42,

    # Mantains the same class proportions
    stratify=y
)

print("\nTraining set:")
print(y_train.value_counts())

print("\nTest set:")
print(y_test.value_counts())


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=20000
)

# TF-IDF learns the vocabulary and IDF
# exclusively from the training data
x_train_tfidf = vectorizer.fit_transform(x_train)

# The test set uses only the vocabulary already learned
x_test_tfidf = vectorizer.transform(x_test)

print("\nTF-IDF matrix dimensions:")
print(x_train_tfidf.shape)


# =========================
# MLP
# =========================

print("\n=========================")
print("MLP")
print("=========================")

mlp_model, mlp_predictions = run_mlp(
    x_train_tfidf,
    y_train,
    x_test_tfidf
)


# =========================
# MLP EVALUATION
# =========================

mlp_accuracy = accuracy_score(
    y_test,
    mlp_predictions
)

mlp_cm = confusion_matrix(
    y_test,
    mlp_predictions
)

print("\nTest accuracy:", mlp_accuracy)

print("\nConfusion Matrix:")
print(mlp_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        mlp_predictions
    )
)


# =========================
# LOGISTIC REGRESSION
# =========================

print("\n=========================")
print("Logistic Regression")
print("=========================")

logistic_model, logistic_predictions = run_logistic_regression(
    x_train_tfidf,
    y_train,
    x_test_tfidf
)


# =========================
# LOGISTIC REGRESSION EVALUATION
# =========================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

print("\nTest accuracy:", logistic_accuracy)

print("\nConfusion Matrix:")
print(logistic_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


# =========================
# SAVING
# =========================

# Save the new MLP model with different name
torch.save(
    mlp_model.state_dict(),
    f"models/phishing_model_mlp_{suffix}.pth"
)

# Save the new Logistic Regression model with a different name
joblib.dump(
    logistic_model,
    f"models/phishing_model_logistic_{suffix}.pkl"
)

# Save the new TF-IDF vectorizer with a different name
joblib.dump(
    vectorizer,
    f"vectorizers/tfidf_vectorizer_{suffix}.pkl"
)

print("\nNew model and TF-IDF vectorizer saved successfully.")
