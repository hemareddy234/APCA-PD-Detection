"""
feature_extraction.py

Multi-channel speech feature extraction for Parkinson's Disease Detection.

Features:
1. Mel Spectrogram
2. Constant-Q Transform (CQT)
3. GFCC-like representation (MFCC)

The extracted features are concatenated into a single feature vector
and saved as a CSV file for subsequent PCA and APCA processing.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from preprocessing import preprocess_audio

# ============================================================
# DATASET PATHS
# ============================================================

DATASET_PATH = "dataset"

PD_FOLDER = os.path.join(DATASET_PATH, "PD")
HC_FOLDER = os.path.join(DATASET_PATH, "HC")

OUTPUT_CSV = "results/italian_pd_multichannel_features_655.csv"

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_multichannel_features(file_path):
    """
    Extract fused multi-channel speech features.

    Parameters
    ----------
    file_path : str
        Path to WAV file.

    Returns
    -------
    numpy.ndarray
        Concatenated feature vector.
    """

    try:

        # ---------------------------------------
        # PREPROCESSING
        # ---------------------------------------

        y, sr = preprocess_audio(file_path)

        # ---------------------------------------
        # FIXED LENGTH (5 Seconds)
        # ---------------------------------------

        target_length = 5 * sr

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # ---------------------------------------
        # MEL SPECTROGRAM
        # ---------------------------------------

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=64
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ---------------------------------------
        # CONSTANT-Q TRANSFORM
        # ---------------------------------------

        cqt = np.abs(
            librosa.cqt(
                y,
                sr=sr,
                n_bins=84
            )
        )

        cqt = librosa.amplitude_to_db(
            cqt,
            ref=np.max
        )

        # ---------------------------------------
        # GFCC-LIKE REPRESENTATION
        # (MFCC Approximation)
        # ---------------------------------------

        gfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        # ---------------------------------------
        # FEATURE FUSION
        # ---------------------------------------

        features = np.concatenate([
            mel.flatten(),
            gfcc.flatten(),
            cqt.flatten()
        ])

        return features

    except Exception as e:

        print(f"Error processing {file_path}")
        print(e)

        return None


# ============================================================
# DATASET PROCESSING
# ============================================================

all_features = []
all_labels = []
all_paths = []


def process_folder(folder_path, label):

    wav_files = []

    for root, _, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    print(f"Processing {len(wav_files)} files from {folder_path}")

    for file_path in tqdm(wav_files):

        features = extract_multichannel_features(file_path)

        if features is not None:

            all_features.append(features)
            all_labels.append(label)
            all_paths.append(file_path)


# ============================================================
# MAIN
# ============================================================

def main():

    process_folder(PD_FOLDER, label=1)
    process_folder(HC_FOLDER, label=0)

    X = np.array(all_features)

    print("Feature Matrix Shape:", X.shape)

    feature_columns = [
        f"f_{i}"
        for i in range(X.shape[1])
    ]

    df = pd.DataFrame(
        X,
        columns=feature_columns
    )

    df["label"] = all_labels
    df["filepath"] = all_paths

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(f"\nFeature CSV saved to: {OUTPUT_CSV}")
    print(df.head())
    print("\nFinal CSV Shape:", df.shape)


if __name__ == "__main__":
    main()
