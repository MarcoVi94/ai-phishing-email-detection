from sklearn.linear_model import LogisticRegression


# =========================
# TRAINING AND PREDICTION
# =========================

def run_logistic_regression(
    x_train_tfidf,
    y_train,
    x_test_tfidf
):

# =========================
# MODEL CREATION
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
# PREDICTION
# =========================

    predictions = model.predict(
        x_test_tfidf
    )


    # Return the model and predictions
    return model, predictions
