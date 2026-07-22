# APCA-PD-Detection

# Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection

This repository contains the official implementation of the proposed **Adaptive Probabilistic Cellular Automata (APCA)** framework for Parkinson's disease detection using multi-channel speech features.

The proposed framework integrates speech preprocessing, multi-channel feature extraction, leakage-free Principal Component Analysis (PCA), Adaptive Probabilistic Cellular Automata (APCA), and a Deep Neural Network (DNN) classifier under subject-independent Group K-Fold cross-validation.

---

# Repository Structure

```
APCA-PD-Detection/
│
├── code/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── cross_validation.py
│   ├── pca.py
│   ├── apca.py
│   ├── experiment.py
│   ├── baseline.py
│   ├── ablation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── data/
│   └── italian_pd_multichannel_features_655.csv
│
├── README.md
├── requirements.txt
└── LICENSE (optional)
```

---

# Methodology

The proposed framework follows the pipeline below:

1. Speech preprocessing
2. Multi-channel feature extraction
   - Mel Spectrogram
   - GFCC
   - Constant-Q Transform (CQT)
3. Subject-independent Group K-Fold cross-validation
4. Leakage-free Standardization
5. Principal Component Analysis (PCA)
6. Adaptive Probabilistic Cellular Automata (APCA)
7. Deep Neural Network (DNN) classification
8. Performance evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

---
---

# Feature Extraction Parameters

The multi-channel speech features are extracted using the implementation provided in `code/feature_extraction.py`.

### Audio Preprocessing

- Speech preprocessing using Wiener filtering (`preprocessing.py`)
- Fixed signal duration: **5 seconds**
- Shorter recordings are zero-padded
- Longer recordings are truncated to 5 seconds

### Mel Spectrogram

- Library: Librosa
- Number of Mel bands (`n_mels`): **64**
- Converted to decibel (dB) scale using `librosa.power_to_db()`

### GFCC-like Representation

- Implemented using MFCC as a GFCC approximation
- Library: Librosa
- Number of coefficients (`n_mfcc`): **40**

### Constant-Q Transform (CQT)

- Library: Librosa
- Number of frequency bins (`n_bins`): **84**
- Converted to decibel (dB) scale using `librosa.amplitude_to_db()`

### Feature Fusion

The Mel Spectrogram, GFCC-like representation, and CQT features are flattened and concatenated into a single feature vector, which is saved as:

```
results/italian_pd_multichannel_features_655.csv
```

---

# Dataset

This work uses the **Italian Parkinson's Voice and Speech Dataset**.

The original speech recordings are publicly available from the original dataset source and are **not redistributed** in this repository.

The extracted feature file used in this work can be downloaded from:

**Google Drive**

[YOUR_GOOGLE_DRIVE_LINK](https://drive.google.com/file/d/15-7JgbBwZcOwK-20NLw6YeZu44TxdJBR/view?usp=sharing)

Download

```
italian_pd_multichannel_features_655.csv
```

Place the downloaded file as:

```
data/
└── italian_pd_multichannel_features_655.csv
```

If you wish to regenerate the feature file from the original WAV recordings, run:

```bash
python code/feature_extraction.py
```

---

# Experimental Settings

The experiments were performed using the following settings:

- Python 3.10
- Subject-independent GroupKFold (5 folds)
- Random Seed = 42
- Standardization fitted only on training folds
- PCA Components = 512
- APCA applied after PCA
- Deep Neural Network implemented in PyTorch
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100
- Loss Function: CrossEntropyLoss

---

# Requirements

Required packages:

- Python 3.10
- PyTorch
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Librosa
- Matplotlib
- tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Code Modules

| Module | Description |
|---------|-------------|
| preprocessing.py | Speech preprocessing using Wiener filtering |
| feature_extraction.py | Multi-channel feature extraction (Mel Spectrogram, GFCC and CQT) |
| cross_validation.py | Subject-independent GroupKFold validation |
| pca.py | Leakage-free standardization and PCA |
| apca.py | Adaptive Probabilistic Cellular Automata implementation |
| experiment.py | Main APCA experiment |
| baseline.py | Baseline model experiments |
| ablation.py | Ablation study |
| model.py | Deep Neural Network architecture |
| train.py | Model training |
| evaluate.py | Performance evaluation |

---

# Reproducing the Results

Download the feature CSV and place it inside the **data/** directory.

Run the following scripts.

## Baseline Experiments

```bash
python code/baseline.py
```

This generates the baseline performance results.

---

## Ablation Study

```bash
python code/ablation.py
```

This generates the ablation study reported in the manuscript.

---

## Proposed APCA Model

```bash
python code/experiment.py
```

This performs:

- Subject-independent GroupKFold
- Leakage-free Standardization
- PCA
- APCA
- DNN Classification
- Performance Evaluation

---

# Reproducibility

To ensure reproducibility:

- Subject-independent GroupKFold is used throughout the experiments.
- Standardization and PCA are fitted only on the training folds to avoid data leakage.
- APCA is applied after PCA transformation.
- Random seeds are fixed.
- The Deep Neural Network is implemented using PyTorch.
- Baseline and ablation experiments are provided separately.

---

# Citation

If you use this implementation in your research, please cite:

```
Hemasudharani et al.

Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection.

(The citation will be updated after publication.)
```

---

# License

This repository is provided for academic and research purposes only.

