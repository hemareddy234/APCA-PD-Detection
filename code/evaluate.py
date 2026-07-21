"""
evaluate.py

Model evaluation utilities for Parkinson's Disease Detection.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate the trained model.

    Parameters
    ----------
    model : nn.Module
        Trained ParkinsonDNN model.

    X_test : ndarray
        Test features.

    y_test : ndarray
        Ground truth labels.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics and predictions.
    """

    model.eval()

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        outputs = model(X_test)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    y_pred = predictions.cpu().numpy()
    y_prob = probabilities.cpu().numpy()

    results = {

        "accuracy": accuracy_score(y_test, y_pred),

        "precision": precision_score(y_test, y_pred),

        "recall": recall_score(y_test, y_pred),

        "f1_score": f1_score(y_test, y_pred),

        "roc_auc": roc_auc_score(y_test, y_prob),

        "confusion_matrix": confusion_matrix(y_test, y_pred),

        "y_pred": y_pred,

        "y_prob": y_prob

    }

    return results


def print_metrics(results):
    """
    Print evaluation metrics.
    """

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"Accuracy      : {results['accuracy']:.4f}")
    print(f"Precision     : {results['precision']:.4f}")
    print(f"Recall        : {results['recall']:.4f}")
    print(f"F1-Score      : {results['f1_score']:.4f}")
    print(f"ROC-AUC       : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
