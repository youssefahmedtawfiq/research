# core/encoding.py
import numpy as np


def threshold_based_encoding(
    features: np.ndarray,
    thresholds: np.ndarray = None,
    mode: str = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary spike encoding:
        s_i(k) = 1 if f_i(k) > theta_i else 0

    Input:
        features shape = (n_windows, n_features)

    Returns:
        spikes shape = (n_windows, n_features)
        thresholds shape = (n_features,)
    """
    if thresholds is None:
        if mode == "median":
            thresholds = np.median(features, axis=0)
        elif mode == "mean":
            thresholds = np.mean(features, axis=0)
        elif mode == "zero":
            thresholds = np.zeros(features.shape[1], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

    spikes = (features > thresholds).astype(np.int32)
    return spikes, thresholds


def minmax_normalize_per_feature(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize each feature into [0, 1].
    Useful before rate encoding.
    """
    f_min = np.min(features, axis=0, keepdims=True)
    f_max = np.max(features, axis=0, keepdims=True)
    return (features - f_min) / (f_max - f_min + eps)
def rate_encoding(normalized_features, time_steps=20, random_seed=None):
    """
   
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # 1. إذا كانت البيانات 3D (عينات، زمن، قنوات) - وهو النظام الجديد بتاعك
    if normalized_features.ndim == 3:
        # هنا الزمن موجود بالفعل، فقوة الإشارة (بين 0 و 1) تمثل احتمالية النبضة
        # سيتم توليد مصفوفة بنفس الأبعاد (عينات، زمن، قنوات) تحتوي على 0 أو 1
        random_matrix = np.random.rand(*normalized_features.shape)
        spikes = (random_matrix < normalized_features).astype(int)
        return spikes
        
    # 2. إذا كانت البيانات 2D (النظام القديم أو الـ Features)
    elif normalized_features.ndim == 2:
        n_windows, n_features = normalized_features.shape
        spikes = np.zeros((n_windows, time_steps, n_features), dtype=int)
        for t in range(time_steps):
            random_matrix = np.random.rand(n_windows, n_features)
            spikes[:, t, :] = (random_matrix < normalized_features).astype(int)
        return spikes
        
    else:
        raise ValueError(f"Unsupported shape for rate_encoding: {normalized_features.shape}")