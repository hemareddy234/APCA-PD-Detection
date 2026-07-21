# APCA-PD-Detection

## Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection

This repository contains the implementation of the proposed **Adaptive Probabilistic Cellular Automata (APCA)** framework for Parkinson's disease detection using multi-channel speech features.

The proposed framework integrates multi-channel speech feature extraction, leakage-free Principal Component Analysis (PCA), Adaptive Probabilistic Cellular Automata (APCA), and a Deep Neural Network (DNN) classifier under subject-independent Group K-Fold cross-validation.

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
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
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

# Dataset

This work uses the **Italian Parkinson's Voice and Speech Dataset**.

The dataset is **not included** in this repository.

Please obtain the dataset from the original source and organize it as follows:

```
dataset/
│
├── HC/
└── PD/
```

---

# Requirements

- Python 3.10 (tested)
- PyTorch
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Librosa
- Matplotlib
- tqdm

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

# Code Modules

| Module | Description |
|---------|-------------|
| preprocessing.py | Speech preprocessing using Wiener filtering |
| feature_extraction.py | Multi-channel feature extraction (Mel Spectrogram, GFCC and CQT) |
| cross_validation.py | Subject-independent Group K-Fold validation |
| pca.py | Leakage-free Standardization and PCA |
| apca.py | Adaptive Probabilistic Cellular Automata implementation |
| model.py | Deep Neural Network architecture |
| train.py | Model training |
| evaluate.py | Performance evaluation |

---

# Notes

- Subject-independent evaluation is performed using Group K-Fold cross-validation.
- Standardization and PCA are fitted only on the training folds to avoid data leakage.
- APCA is applied after PCA feature transformation.
- The Deep Neural Network classifier is implemented using PyTorch.
- Random seed is fixed to ensure reproducible experiments.

---

# Citation

If you use this implementation in your research, please cite the corresponding publication.

```
Hemasudharani et al.,
"Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection."

(The citation will be updated after publication.)
```

---

# License

This repository is provided for academic and research purposes.

