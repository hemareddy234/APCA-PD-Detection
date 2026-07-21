"""
baseline.py

Baseline comparison using pretrained speech embeddings.

Methods:
1. HuBERT + DNN
2. WavLM + DNN
3. Wav2Vec2 + DNN
"""

import os
import re
import pandas as pd

from experiment import run_experiment


# ==========================================================
# SUBJECT EXTRACTION
# ==========================================================

def extract_subject(filename):

    filename = os.path.basename(str(filename))
    filename = filename.replace(".wav", "")

    subject = re.sub(
        r'^(D1|D2|B1|B2|FB1|PR1|VA1|VA2|VE1|VE2|VI1|VI2|VO1|VO2|VU1|VU2)',
        '',
        filename
    )

    return subject


# ==========================================================
# LOAD FEATURES
# ==========================================================

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    df["subject"] = df["filename"].apply(extract_subject)

    X = df.drop(
        columns=[
            "filename",
            "label",
            "subject"
        ]
    ).values.astype("float32")

    y = df["label"].values

    subjects = df["subject"]

    return X, y, subjects


# ==========================================================
# CHANGE THESE PATHS
# ==========================================================

HUBERT_CSV = "/content/drive/MyDrive/hubert_embeddings.csv"

WAVLM_CSV = "/content/drive/MyDrive/wavlm_embeddings.csv"

WAV2VEC2_CSV = "/content/drive/MyDrive/wav2vec2_embeddings.csv"

# ==========================================================
# HuBERT + DNN
# ==========================================================

print("\n" + "=" * 70)
print("HuBERT + DNN")
print("=" * 70)

X, y, subjects = load_dataset(HUBERT_CSV)

hubert_acc, hubert_std, _, _, _ = run_experiment(
    X,
    y,
    subjects,
    use_pca=False,
    use_apca=False
)


# ==========================================================
# WavLM + DNN
# ==========================================================

print("\n" + "=" * 70)
print("WavLM + DNN")
print("=" * 70)

X, y, subjects = load_dataset(WAVLM_CSV)

wavlm_acc, wavlm_std, _, _, _ = run_experiment(
    X,
    y,
    subjects,
    use_pca=False,
    use_apca=False
)


# ==========================================================
# Wav2Vec2 + DNN
# ==========================================================

print("\n" + "=" * 70)
print("Wav2Vec2 + DNN")
print("=" * 70)

X, y, subjects = load_dataset(WAV2VEC2_CSV)

wav2vec_acc, wav2vec_std, _, _, _ = run_experiment(
    X,
    y,
    subjects,
    use_pca=False,
    use_apca=False
)


# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame({

    "Method": [

        "HuBERT + DNN",

        "WavLM + DNN",

        "Wav2Vec2 + DNN"

    ],

    "Accuracy (%)": [

        hubert_acc,

        wavlm_acc,

        wav2vec_acc

    ],

    "Std (%)": [

        hubert_std,

        wavlm_std,

        wav2vec_std

    ]

})

print("\n")
print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)
print(results)

results.to_csv(
    "baseline_results.csv",
    index=False
)

print("\nResults saved to baseline_results.csv")
