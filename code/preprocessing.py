"""
preprocessing.py

Audio preprocessing functions for Parkinson's disease detection.
"""

import librosa
from scipy.signal import wiener


def load_audio(audio_path, sr=16000):
    """
    Load an audio file.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr


def apply_wiener_filter(y, kernel_size=29):
    """
    Apply Wiener filtering.
    """
    return wiener(y, mysize=kernel_size)


def preprocess_audio(audio_path, sr=16000):
    """
    Load and preprocess an audio file.
    """
    y, sr = load_audio(audio_path, sr)
    y = apply_wiener_filter(y)
    return y, sr
