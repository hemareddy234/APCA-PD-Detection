"""
apca.py

Adaptive Probabilistic Cellular Automata (APCA)

This module implements the proposed Adaptive Probabilistic Cellular
Automata (APCA) used for feature enhancement in Parkinson's disease
detection.
"""

import numpy as np


def apca(grid, iterations=4):
    """
    Apply Adaptive Probabilistic Cellular Automata (APCA).

    Parameters
    ----------
    grid : ndarray
        2D feature grid (32 × 16).

    iterations : int, optional
        Number of APCA iterations (default = 4).

    Returns
    -------
    ndarray
        APCA-enhanced feature grid.
    """

    grid = grid.copy()

    for _ in range(iterations):

        new_grid = grid.copy()

        # Moore Neighborhood
        for i in range(1, 31):
            for j in range(1, 15):

                neighborhood = grid[i-1:i+2, j-1:j+2]

                center = grid[i, j]

                mean_neighbor = np.mean(neighborhood)

                std_neighbor = np.std(neighborhood)

                adaptive_factor = np.tanh(std_neighbor)

                new_grid[i, j] = (
                    0.50 * center +
                    0.30 * mean_neighbor +
                    0.20 * adaptive_factor
                )

        grid = new_grid

    return grid


def apply_apca(fold_data, iterations=4):
    """
    Apply APCA to all GroupKFold splits.

    Parameters
    ----------
    fold_data : list
        PCA-transformed data for each fold.

    iterations : int
        Number of APCA iterations.

    Returns
    -------
    list
        Updated fold_data containing APCA features.
    """

    print("=" * 70)
    print("ADAPTIVE PROBABILISTIC CELLULAR AUTOMATA")
    print("=" * 70)

    for fold in range(len(fold_data)):

        X_train = fold_data[fold]["X_train"]
        X_test = fold_data[fold]["X_test"]

        # 512 → 32 × 16
        X_train_grid = X_train.reshape(-1, 32, 16)
        X_test_grid = X_test.reshape(-1, 32, 16)

        print(f"\nFold {fold + 1}")
        print("-" * 50)

        print("Train Grid :", X_train_grid.shape)
        print("Test Grid  :", X_test_grid.shape)

        X_train_apca = np.array(
            [apca(sample, iterations).flatten()
             for sample in X_train_grid],
            dtype=np.float32
        )

        X_test_apca = np.array(
            [apca(sample, iterations).flatten()
             for sample in X_test_grid],
            dtype=np.float32
        )

        print("Train APCA :", X_train_apca.shape)
        print("Test APCA  :", X_test_apca.shape)

        fold_data[fold]["X_train_apca"] = X_train_apca
        fold_data[fold]["X_test_apca"] = X_test_apca

    return fold_data


if __name__ == "__main__":
    print("APCA module loaded successfully.")
