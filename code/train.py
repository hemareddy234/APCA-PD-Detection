"""
train.py

Training utilities for the Parkinson's Disease Detection model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

SEED = 42


def set_seed(seed=SEED):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model,
    X_train,
    y_train,
    device,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4,
):
    """
    Train the ParkinsonDNN model.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    X_train : ndarray
        Training features.

    y_train : ndarray
        Training labels.

    device : torch.device
        CPU or CUDA device.

    epochs : int
        Number of training epochs.

    batch_size : int
        Mini-batch size.

    learning_rate : float
        Adam learning rate.

    weight_decay : float
        L2 regularization parameter.

    Returns
    -------
    nn.Module
        Best-performing trained model.
    """

    set_seed()

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_acc = 0.0
    best_state = None

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            outputs = model(xb)

            loss = criterion(outputs, yb)

            loss.backward()

            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == yb).sum().item()

            total += yb.size(0)

        train_accuracy = 100.0 * correct / total

        if train_accuracy > best_acc:
            best_acc = train_accuracy
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Accuracy: {train_accuracy:.2f}%"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest Training Accuracy: {best_acc:.2f}%")

    return model


if __name__ == "__main__":
    print("Training module loaded successfully.")
