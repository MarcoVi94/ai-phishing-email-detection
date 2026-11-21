import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# =========================
# MLP MODEL
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
# TRAINING AND PREDICTION
# =========================

def run_mlp(
    x_train_tfidf,
    y_train,
    x_test_tfidf
):

# =========================
# TENSOR CONVERSION
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
# DATASET AND DATALOADER
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
# NETWORK CONSTRUCTION
# =========================

    input_size = x_train_tensor.shape[1]

    model = PhishingClassifier(
        input_size
    )

    print("\nModel:")
    print(model)


# =========================
# LOSS FUNCTION AND OPTIMIZER
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

            # Prediction
            prediction = model(x_batch)

            # Calculate the loss
            loss = loss_function(
                prediction.squeeze(),
                y_batch
            )

            # Reset gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()


        # Average loss per epoch
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


    # Return the model and predictions
    return model, test_predictions.numpy()
