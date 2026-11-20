import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# =========================
# MODELLO MLP
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
# TRAINING E PREDIZIONE
# =========================

def run_mlp(
    x_train_tfidf,
    y_train,
    x_test_tfidf
):

# =========================
# CONVERSIONE IN TENSORI
# =========================

    x_train_tensor = torch.tensor(
        x_train_tfidf.toarray(),
        dtype=torch.float32
    )

    x_test_tensor = torch.tensor(
        x_test_tfidf.toarray(),
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )


# =========================
# DATASET E DATALOADER
# =========================

    train_dataset = TensorDataset(
        x_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True
    )


# =========================
# COSTRUZIONE DELLA RETE
# =========================

    input_size = x_train_tensor.shape[1]

    model = PhishingClassifier(
        input_size
    )

    print("\nModello:")
    print(model)


# =========================
# LOSS E OPTIMIZER
# =========================

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


# =========================
# TRAINING
# =========================

    epochs = 10

    for epoch in range(epochs):

        total_loss = 0

        for x_batch, y_batch in train_loader:

            # Previsione
            prediction = model(x_batch)

            # Calcolo della loss
            loss = loss_function(
                prediction.squeeze(),
                y_batch
            )

            # Azzero i gradienti
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Aggiorno i pesi
            optimizer.step()

            total_loss += loss.item()


        # Loss media dell'epoca
        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"- Loss: {average_loss:.4f}"
        )


# =========================
# TESTING
# =========================

    model.eval()

    with torch.no_grad():

        test_outputs = model(
            x_test_tensor
        )

        # 1 = phishing
        # 0 = legitimate
        test_predictions = (
            test_outputs.squeeze() > 0
        ).float()


    # Restituisco il modello e le predizioni
    return model, test_predictions.numpy()