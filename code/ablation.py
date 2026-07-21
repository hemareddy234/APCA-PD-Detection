"""
ablation.py

Ablation Study

1. Mel + DNN
2. GFCC + DNN
3. CQT + DNN
4. PCA + DNN
"""

import os
import re
import numpy as np
import pandas as pd

from experiment import run_experiment

# ==========================================================
# LOAD DATASET
# ==========================================================

CSV_PATH = "/content/drive/MyDrive/italian_pd_multichannel_features_655.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DATASET LOADED")
print("=" * 70)
print("Dataset Shape :", df.shape)

# ==========================================================
# PREPARE FEATURES
# ==========================================================

feature_cols = [c for c in df.columns if c.startswith("f_")]

X = df[feature_cols].values.astype(np.float32)

y = df["label"].values.astype(np.int64)

filepaths = df["filepath"].values

print("\nFeature Matrix :", X.shape)
print("Labels         :", y.shape)

# ==========================================================
# SUBJECT EXTRACTION
# ==========================================================

def extract_subject(filepath):

    filename = os.path.basename(str(filepath))
    filename = filename.replace(".wav", "")

    subject = re.sub(
        r'^(D1|D2|B1|B2|FB1|PR1|VA1|VA2|VE1|VE2|VI1|VI2|VO1|VO2|VU1|VU2)',
        '',
        filename
    )

    return subject


subjects = np.array([extract_subject(f) for f in filepaths])

# ==========================================================
# FEATURE SPLITTING
# ==========================================================

MEL_DIM = 12800
GFCC_DIM = 256
CQT_DIM = 16460

X_mel = X[:, :MEL_DIM]

X_gfcc = X[:, MEL_DIM:MEL_DIM + GFCC_DIM]

X_cqt = X[:, MEL_DIM + GFCC_DIM:]

X_fused = np.concatenate(
    [X_mel, X_gfcc, X_cqt],
    axis=1
)

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("Mel   :", X_mel.shape)
print("GFCC  :", X_gfcc.shape)
print("CQT   :", X_cqt.shape)
print("Fusion:", X_fused.shape)

# ==========================================================
# MEL + DNN
# ==========================================================

print("\n" + "=" * 70)
print("Mel + DNN")
print("=" * 70)

mel_acc, mel_std, _, _, _ = run_experiment(
    X=X_mel,
    y=y,
    subjects=subjects,
    use_pca=False,
    use_apca=False
)

# ==========================================================
# GFCC + DNN
# ==========================================================

print("\n" + "=" * 70)
print("GFCC + DNN")
print("=" * 70)

gfcc_acc, gfcc_std, _, _, _ = run_experiment(
    X=X_gfcc,
    y=y,
    subjects=subjects,
    use_pca=False,
    use_apca=False
)

# ==========================================================
# CQT + DNN
# ==========================================================

print("\n" + "=" * 70)
print("CQT + DNN")
print("=" * 70)

cqt_acc, cqt_std, _, _, _ = run_experiment(
    X=X_cqt,
    y=y,
    subjects=subjects,
    use_pca=False,
    use_apca=False
)

# ==========================================================
# PCA + DNN
# ==========================================================

print("\n" + "=" * 70)
print("PCA + DNN")
print("=" * 70)

pca_acc, pca_std, pca_true, pca_pred, pca_prob = run_experiment(
    X=X_fused,
    y=y,
    subjects=subjects,
    use_pca=True,
    use_apca=False
)

# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame({

    "Method": [
        "Mel + DNN",
        "GFCC + DNN",
        "CQT + DNN",
        "PCA + DNN"
    ],

    "Accuracy (%)": [
        mel_acc,
        gfcc_acc,
        cqt_acc,
        pca_acc
    ],

    "Std (%)": [
        mel_std,
        gfcc_std,
        cqt_std,
        pca_std
    ]

})

print("\n" + "=" * 70)
print("ABLATION RESULTS")
print("=" * 70)

print(results)

results.to_csv(
    "ablation_results.csv",
    index=False
)

print("\nResults saved to ablation_results.csv")
