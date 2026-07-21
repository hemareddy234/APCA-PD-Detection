"""
pca.py

Leakage-free standardization and
Principal Component Analysis (PCA).
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def perform_pca(X_fused, y, subjects, gkf, n_components=512):
    """
    Perform leakage-free StandardScaler and PCA
    within each Group K-Fold split.

    Parameters
    ----------
    X_fused : ndarray
        Input feature matrix.

    y : ndarray
        Labels.

    subjects : ndarray
        Subject IDs.

    gkf : GroupKFold
        Group K-Fold object.

    n_components : int
        Number of PCA components.

    Returns
    -------
    fold_data : list
        PCA-transformed data for each fold.
    """

    print("=" * 70)
    print("LEAKAGE-FREE STANDARDIZATION + PCA")
    print("=" * 70)

    fold_data = []

    for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X_fused, y, groups=subjects), 1):

        print(f"\nFold {fold}")
        print("-" * 50)

        # ------------------------
        # Train/Test Split
        # ------------------------

        X_train = X_fused[train_idx]
        X_test = X_fused[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ------------------------
        # Standardization
        # ------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ------------------------
        # PCA
        # ------------------------

        pca = PCA(
            n_components=n_components,
            random_state=42
        )

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print("Train PCA Shape :", X_train.shape)
        print("Test PCA Shape  :", X_test.shape)

        print(
            "Explained Variance : %.4f"
            % np.sum(pca.explained_variance_ratio_)
        )

        fold_data.append({

            "X_train": X_train.astype(np.float32),

            "X_test": X_test.astype(np.float32),

            "y_train": y_train,

            "y_test": y_test

        })

    return fold_data
