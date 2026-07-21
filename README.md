# APCA-PD-Detection

## Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection

This repository contains the implementation of the proposed **Adaptive Probabilistic Cellular Automata (APCA)** framework for Parkinson's disease detection using multi-channel speech features.

The proposed pipeline combines multi-channel acoustic feature extraction, leakage-free dimensionality reduction using Principal Component Analysis (PCA), Adaptive Probabilistic Cellular Automata (APCA) feature enhancement, and a Deep Neural Network (DNN) classifier under subject-independent Group K-Fold cross-validation.

---

## Repository Structure

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

## Methodology

The implementation follows the workflow below:

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

## Dataset

This work uses the **Italian Parkinson's Voice and Speech Dataset**.

The dataset is **not included** in this repository. Please obtain it from the original source and organize it as:


dataset/
│
├── PD/
└── HC/
```

---

## Requirements

- Python 3.10 (tested)
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Librosa
- SciPy
- Matplotlib
- tqdm

Install all dependencies using:

bash
pip install -r requirements.txt


---

## Code Modules

| File | Description |
|------|-------------|
| preprocessing.py | Speech preprocessing |
| feature_extraction.py | Multi-channel feature extraction |
| cross_validation.py | Subject-independent Group K-Fold |
| pca.py | Leakage-free PCA |
| apca.py | Adaptive Probabilistic Cellular Automata |
| model.py | Deep Neural Network |
| train.py | Model training |
| evaluate.py | Performance evaluation |

---



## Citation

If you use this code in your research, please cite the corresponding publication:

Hemasudharani et al., "Adaptive Probabilistic Cellular Automata with Multi-Channel Speech Features for Parkinson's Disease Detection", 2026.
```

(Update the citation after your paper is published.)
## Notes

- Subject-independent evaluation is performed using Group K-Fold cross-validation.
- Standardization and PCA are fitted only on the training folds to prevent data leakage.
- APCA is applied after PCA feature transformation.
- The DNN classifier is implemented using PyTorch.
## License

This repository is provided for academic and research purposes.
