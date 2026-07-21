"""
model.py

Deep Neural Network (DNN) classifier used for Parkinson's Disease
detection after APCA feature enhancement.
"""

import torch
import torch.nn as nn


def get_device():
    """
    Return the available computation device.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class ParkinsonDNN(nn.Module):
    """
    Deep Neural Network for binary classification.

    Architecture:
        Input
          ↓
        Linear(256)
          ↓
        BatchNorm
          ↓
        ReLU
          ↓
        Dropout(0.30)
          ↓
        Linear(128)
          ↓
        BatchNorm
          ↓
        ReLU
          ↓
        Dropout(0.30)
          ↓
        Linear(64)
          ↓
        ReLU
          ↓
        Linear(2)
    """

    def __init__(self, input_dim):

        super(ParkinsonDNN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 2)

        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = ParkinsonDNN(input_dim=512)

    print(model)

    print("\nModel created successfully.")
