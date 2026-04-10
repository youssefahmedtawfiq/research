# core/split.py
from typing import Tuple

import numpy as np


def segment_signal(
    emg: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    repetitions: np.ndarray = None,
    exclude_rest_label: bool = False,
    rest_label: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment EMG into overlapping windows and assign one label per window.

    Window label is chosen as the center sample label.
    """
    segments = []
    window_labels = []
    window_repetitions = []

    n_samples = emg.shape[0]

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        center = start + window_size // 2

        label = int(labels[center])

        if exclude_rest_label and label == rest_label:
            continue

        segments.append(emg[start:end])
        window_labels.append(label)

        if repetitions is not None:
            window_repetitions.append(int(repetitions[center]))
        else:
            window_repetitions.append(-1)

    return (
        np.asarray(segments, dtype=np.float64),
        np.asarray(window_labels, dtype=np.int32),
        np.asarray(window_repetitions, dtype=np.int32),
    )