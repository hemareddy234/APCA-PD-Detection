"""
experiment.py

Generic Experiment Pipeline for Parkinson's Disease Detection

This module performs:
1. GroupKFold Cross-Validation
2. Standardization
3. PCA (Optional)
4. APCA (Optional)
5. DNN Training
6. Model Evaluation
"""

import numpy as np
import torch

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from model import ParkinsonDNN
from train import train_model
from evaluate import evaluate_model
from apca import apca


# ==========================================================
# SETTINGS
# ==========================================================

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# GENERIC EXPERIMENT FUNCTION
# ==========================================================

def run_experiment(
    X,
    y,
    subjects,
    use_pca=False,
    use_apca=False
):

    gkf = GroupKFold(n_splits=5)

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    auc_list = []

    all_true = []
    all_pred = []
    all_prob = []

    for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=subjects), 1):

        print("\n" + "=" * 60)
        print(f"Fold {fold}")
        print("=" * 60)

        # --------------------------------------------------
        # Train / Test Split
        # --------------------------------------------------

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # --------------------------------------------------
        # Standardization
        # --------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        if use_pca:

            pca = PCA(
                n_components=512,
                random_state=SEED
            )

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # --------------------------------------------------
        # APCA
        # --------------------------------------------------

        if use_apca:

            X_train_grid = X_train.reshape(-1, 32, 16)
            X_test_grid = X_test.reshape(-1, 32, 16)

            X_train = np.array(
                [apca(x).flatten() for x in X_train_grid],
                dtype=np.float32
            )

            X_test = np.array(
                [apca(x).flatten() for x in X_test_grid],
                dtype=np.float32
            )

        # --------------------------------------------------
        # Build DNN
        # --------------------------------------------------

        model = ParkinsonDNN(
            input_dim=X_train.shape[1]
        ).to(device)

        # --------------------------------------------------
        # Train Model
        # --------------------------------------------------

        model = train_model(
            model,
            X_train,
            y_train
        )

        # --------------------------------------------------
        # Evaluate Model
        # --------------------------------------------------

        results = evaluate_model(
            model,
            X_test,
            y_test,
            device
        )

        acc = results["accuracy"]
        pre = results["precision"]
        rec = results["recall"]
        f1 = results["f1_score"]
        auc = results["roc_auc"]

        y_pred = results["y_pred"]
        y_prob = results["y_prob"]

        print(f"Accuracy : {acc * 100:.2f}%")

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        auc_list.append(auc)

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

    # ======================================================
    # Final Results
    # ======================================================

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Accuracy : {np.mean(acc_list) * 100:.2f} ± {np.std(acc_list) * 100:.2f}")
    print(f"Precision: {np.mean(pre_list) * 100:.2f}")
    print(f"Recall   : {np.mean(rec_list) * 100:.2f}")
    print(f"F1 Score : {np.mean(f1_list) * 100:.2f}")
    print(f"ROC-AUC  : {np.mean(auc_list) * 100:.2f}")

    return (
        np.mean(acc_list) * 100,
        np.std(acc_list) * 100,
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob)
    )


if __name__ == "__main__":
    print("Experiment module loaded successfully.")
