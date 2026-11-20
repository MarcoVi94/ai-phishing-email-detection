import torch
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from mlp import run_mlp
from logistic_regression import run_logistic_regression


# =========================
# CARICAMENTO DATASET
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
    # il testo dell'email è già contenuto
    # nella colonna text_combined
    df["text_combined"] = df["text_combined"].fillna("")


elif DATASET == 2:

    # Dataset 2:
    # unisco subject e body per creare
    # il testo completo dell'email
    df["text_combined"] = (
        df["subject"].fillna("") + " " +
        df["body"].fillna("")
    )

# Elimino i duplicati
df = df.drop_duplicates(subset="text_combined").copy()

print("Dimensioni dopo la rimozione dei duplicati:", df.shape)


# =========================
# INPUT E LABEL
# =========================

# X contiene le email
# y contiene le classificazioni
x = df["text_combined"]
y = df["label"]


# =========================
# DIVISIONE TRAIN / TEST
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    # il 20% sarà usato per il test
    test_size=0.20,

    # la divisione sarà sempre la stessa
    random_state=42,

    # mantiene la stessa proporzione delle classi
    stratify=y
)

print("\nTrain:")
print(y_train.value_counts())

print("\nTest:")
print(y_test.value_counts())


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=20000
)

# Il TF-IDF impara il vocabolario e l'IDF
# esclusivamente sui dati di training
x_train_tfidf = vectorizer.fit_transform(x_train)

# Sul test viene utilizzato solamente il vocabolario già appreso
x_test_tfidf = vectorizer.transform(x_test)

print("\nDimensione matrice TF-IDF:")
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
# VALUTAZIONE MLP
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
# VALUTAZIONE LOGISTIC REGRESSION
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
# SALVATAGGIO
# =========================

# Salvo il nuovo modello MLP con un nome diverso
torch.save(
    mlp_model.state_dict(),
    f"models/phishing_model_mlp_{suffix}.pth"
)

# Salvo il nuovo modello Logistic Regression con un nome diverso
joblib.dump(
    logistic_model,
    f"models/phishing_model_logistic_{suffix}.pkl"
)

# Salvo il nuovo TF-IDF con un nome diverso
joblib.dump(
    vectorizer,
    f"vectorizers/tfidf_vectorizer_{suffix}.pkl"
)

print("\nNuovi modelli e TF-IDF salvati correttamente.")