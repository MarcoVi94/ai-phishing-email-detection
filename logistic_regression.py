from sklearn.linear_model import LogisticRegression


# =========================
# TRAINING E PREDIZIONE
# =========================

def run_logistic_regression(
    x_train_tfidf,
    y_train,
    x_test_tfidf
):

# =========================
# CREAZIONE DEL MODELLO
# =========================

    model = LogisticRegression(
        max_iter=1000
    )


# =========================
# TRAINING
# =========================

    model.fit(
        x_train_tfidf,
        y_train
    )


# =========================
# PREDIZIONE
# =========================

    predictions = model.predict(
        x_test_tfidf
    )


    # Restituisco il modello e le predizioni
    return model, predictions