import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ========================
# Percorsi
# ========================

DATASET1 = "datasets/phishing_email.csv"
DATASET2 = "datasets/AVN_Corpus.csv"

VECTORIZER1 = "vectorizers/tfidf_vectorizer_dataset1.pkl"
VECTORIZER2 = "vectorizers/tfidf_vectorizer_dataset2.pkl"

MLP1 = "models/phishing_model_mlp_dataset1.pth"
MLP2 = "models/phishing_model_mlp_dataset2.pth"

LOGISTIC1 = "models/phishing_model_logistic_dataset1.pkl"
LOGISTIC2 = "models/phishing_model_logistic_dataset2.pkl"


# ========================
# Modello MLP
# ========================


class MLP(nn.Module):

    def __init__(self, input_size=20000):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


def load_mlp(model_path):

    model = MLP(input_size=20000)

    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ========================
# Caricamento Dataset 1
# ========================

def load_dataset1():

    df = pd.read_csv(DATASET1)

    df = df.fillna("")

    df = df.drop_duplicates(
        subset="text_combined"
    ).copy()

    x = df["text_combined"].astype(str)
    y = df["label"].astype(int)

    return x, y


# ========================
# Caricamento Dataset 2
# ========================

def load_dataset2():

    df = pd.read_csv(DATASET2)

    df = df.fillna("")

    df["text_combined"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    )

    df = df.drop_duplicates(
        subset="text_combined"
    ).copy()

    x = df["text_combined"].astype(str)
    y = df["label"].astype(int)

    return x, y


# ========================
# Predizioni MLP
# ========================

def predict_mlp(model, x_vectorized):

    x_tensor = torch.tensor(
        x_vectorized.toarray(),
        dtype=torch.float32
    )

    with torch.no_grad():

        logits = model(x_tensor)

        probabilities = torch.sigmoid(
            logits
        ).squeeze()

    predictions = (
        probabilities >= 0.5
    ).int().numpy()

    return predictions


# ========================
# Esecuzione esperimento
# ========================

def run_experiment(
    model_name,
    model,
    vectorizer,
    x_test,
    y_test,
    experiment_name
):

    # Trasformazione usando il vettorizzatore
    # appartenente al dataset di training
    x_vectorized = vectorizer.transform(x_test)

    # Predizioni
    if model_name == "MLP":

        predictions = predict_mlp(
            model,
            x_vectorized
        )

    else:

        predictions = model.predict(
            x_vectorized
        )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Matrice di confusione
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print(experiment_name)
    print("=" * 60)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nMatrice di confusione:")
    print(cm)

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legittima",
                "Phishing"
            ],
            digits=4
        )
    )

    return {
        "Esperimento": experiment_name,
        "Accuracy": accuracy * 100,
        "TN": cm[0][0],
        "FP": cm[0][1],
        "FN": cm[1][0],
        "TP": cm[1][1]
    }


# ========================
# MAIN
# ========================

def main():

    print("=" * 60)
    print("CROSS-DATASET EXPERIMENTS")
    print("=" * 60)

    # --------------------
    # Dataset
    # --------------------

    print("\nCaricamento dataset...")

    x1, y1 = load_dataset1()
    x2, y2 = load_dataset2()

    print(
        f"Dataset 1: {len(x1)} email"
    )

    print(
        f"Dataset 2: {len(x2)} email"
    )

    # --------------------
    # Vettorizzatori
    # --------------------

    print("\nCaricamento vettorizzatori...")

    vectorizer1 = joblib.load(
        VECTORIZER1
    )

    vectorizer2 = joblib.load(
        VECTORIZER2
    )

    print("Vettorizzatori caricati.")

    # --------------------
    # Logistic Regression
    # --------------------

    print(
        "\nCaricamento modelli Logistic Regression..."
    )

    logistic1 = joblib.load(
        LOGISTIC1
    )

    logistic2 = joblib.load(
        LOGISTIC2
    )

    print(
        "Modelli Logistic Regression caricati."
    )

    # --------------------
    # MLP
    # --------------------

    print("\nCaricamento modelli MLP...")

    mlp1 = load_mlp(MLP1)
    mlp2 = load_mlp(MLP2)

    print("Modelli MLP caricati.")

    # --------------------
    # Risultati
    # --------------------

    results = []

    # ==============================
    # 1. MLP Dataset 1 -> Dataset 2
    # ==============================

    results.append(
        run_experiment(
            "MLP",
            mlp1,
            vectorizer1,
            x2,
            y2,
            "MLP - Dataset 1 -> Dataset 2"
        )
    )

    # ===============================
    # 2. MLP Dataset 2 -> Dataset 1
    # ===============================

    results.append(
        run_experiment(
            "MLP",
            mlp2,
            vectorizer2,
            x1,
            y1,
            "MLP - Dataset 2 -> Dataset 1"
        )
    )

    # ===============================
    # 3. Logistic Dataset 1 -> Dataset 2
    # ===============================

    results.append(
        run_experiment(
            "Logistic Regression",
            logistic1,
            vectorizer1,
            x2,
            y2,
            "Logistic Regression - Dataset 1 -> Dataset 2"
        )
    )

    # ===============================
    # 4. Logistic Dataset 2 -> Dataset 1
    # ===============================

    results.append(
        run_experiment(
            "Logistic Regression",
            logistic2,
            vectorizer2,
            x1,
            y1,
            "Logistic Regression - Dataset 2 -> Dataset 1"
        )
    )

    # ===============================
    # Riepilogo
    # ===============================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("RIEPILOGO CROSS-DATASET")
    print("=" * 60)

    print(
        results_df[
            [
                "Esperimento",
                "Accuracy",
                "TN",
                "FP",
                "FN",
                "TP"
            ]
        ].to_string(index=False)
    )

    # -------------------------------
    # Salvataggio risultati
    # -------------------------------

    output_file = "cross_dataset_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nRisultati salvati in: {output_file}"
    )


# ===================================
# Avvio
# ===================================

if __name__ == "__main__":
    main()