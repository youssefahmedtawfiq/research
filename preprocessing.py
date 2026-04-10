# core/preprocessing.py
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


def resample_emg(emg: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """
    Resample EMG from orig_fs to target_fs.
    """
    if orig_fs == target_fs:
        return emg

    from math import gcd
    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(emg, up, down, axis=0)


def butter_filter(
    signal: np.ndarray,
    cutoff,
    fs: int,
    btype: str,
    order: int = 4
) -> np.ndarray:
    nyquist = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyquist for c in cutoff]
    else:
        wn = cutoff / nyquist

    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, notch_freq: float, fs: int, q: float = 30.0) -> np.ndarray:
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal, axis=0)


def preprocess_emg(
    emg: np.ndarray,
    fs: int,
    highpass_hz: float = 20.0,
    notch_hz: float = 50.0,
    lowpass_hz: float = 450.0,
    filter_order: int = 4
) -> np.ndarray:
    """
    EMG preprocessing pipeline:
    1) High-pass filter
    2) Notch filter
    3) Low-pass filter
    """
    x = butter_filter(emg, cutoff=highpass_hz, fs=fs, btype="highpass", order=filter_order)
    x = notch_filter(x, notch_freq=notch_hz, fs=fs)
    x = butter_filter(x, cutoff=lowpass_hz, fs=fs, btype="lowpass", order=filter_order)
    return x


def sliding_window_zscore(
    emg: np.ndarray,
    window_size_samples: int,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Sliding-window z-score normalization:
        x_norm = (x - mu_w) / sigma_w

    Computes a local mean/std around each sample using a moving window.
    """
    n_samples, n_channels = emg.shape
    half = window_size_samples // 2
    normalized = np.zeros_like(emg, dtype=np.float64)

    for ch in range(n_channels):
        x = emg[:, ch]
        for i in range(n_samples):
            start = max(0, i - half)
            end = min(n_samples, i + half)
            window = x[start:end]
            mu = np.mean(window)
            sigma = np.std(window)
            normalized[i, ch] = (x[i] - mu) / (sigma + eps)

    return normalized