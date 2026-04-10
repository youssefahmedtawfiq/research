# core/features.py
import numpy as np


def mean_absolute_value(x: np.ndarray) -> float:
    return np.mean(np.abs(x))


def waveform_length(x: np.ndarray) -> float:
    return np.sum(np.abs(np.diff(x)))


def variance_feature(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return np.var(x, ddof=1)


def zero_crossings(x: np.ndarray, threshold: float = 1e-3) -> int:
    count = 0
    for i in range(len(x) - 1):
        if (x[i] * x[i + 1] < 0) and (abs(x[i] - x[i + 1]) > threshold):
            count += 1
    return count


def slope_sign_changes(x: np.ndarray, threshold: float = 1e-3) -> int:
    count = 0
    for i in range(1, len(x) - 1):
        diff1 = x[i] - x[i - 1]
        diff2 = x[i] - x[i + 1]
        if (diff1 * diff2 > 0) and (abs(diff1) > threshold or abs(diff2) > threshold):
            count += 1
    return count


def extract_features_from_window(
    window: np.ndarray,
    zc_threshold: float = 1e-3,
    ssc_threshold: float = 1e-3
) -> np.ndarray:
    """
    Extract features from one EMG window.

    Input:
        window shape = (window_size, n_channels)

    Output:
        feature_matrix shape = (n_channels, 5)
        Feature order = [MAV, WL, VAR, ZC, SSC]
    """
    n_channels = window.shape[1]
    features = np.zeros((n_channels, 5), dtype=np.float64)

    for ch in range(n_channels):
        x = window[:, ch]
        features[ch, 0] = mean_absolute_value(x)
        features[ch, 1] = waveform_length(x)
        features[ch, 2] = variance_feature(x)
        features[ch, 3] = zero_crossings(x, threshold=zc_threshold)
        features[ch, 4] = slope_sign_changes(x, threshold=ssc_threshold)

    return features


def extract_features_from_segments(
    segments: np.ndarray,
    zc_threshold: float = 1e-3,
    ssc_threshold: float = 1e-3
) -> np.ndarray:
    """
    Extract features from all segments.

    Input:
        segments shape = (n_windows, window_size, n_channels)

    Output:
        all_features shape = (n_windows, n_channels, 5)
    """
    all_features = []
    for window in segments:
        feats = extract_features_from_window(
            window,
            zc_threshold=zc_threshold,
            ssc_threshold=ssc_threshold
        )
        all_features.append(feats)

    return np.asarray(all_features, dtype=np.float64)


def flatten_feature_matrix(feature_tensor: np.ndarray) -> np.ndarray:
    """
    Flatten (n_windows, n_channels, n_features)
    into   (n_windows, n_channels * n_features)
    """
    n_windows = feature_tensor.shape[0]
    return feature_tensor.reshape(n_windows, -1)


def feature_zscore_normalize(X: np.ndarray, eps: float = 1e-8):
    """
    Normalize features column-wise.
    """
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    Xn = (X - mu) / (sigma + eps)
    return Xn, mu, sigma