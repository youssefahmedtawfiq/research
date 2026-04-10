# core/dataset.py
import os
from typing import Optional, Tuple
import numpy as np
import scipy.io as sio


def _get_first_existing_key(data: dict, candidate_keys):
    for key in candidate_keys:
        if key in data:
            return data[key]
    return None


def load_ninapro_db2(
    subject: int = 1,
    exercise: int = 2,
    acquisition: int = 1,
    data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    """
    Load Ninapro DB2 .mat file.

    Expected file name pattern:
        S{subject}_E{exercise}_A{acquisition}.mat

    Returns:
        emg: ndarray of shape (n_samples, n_channels)
        labels: ndarray of shape (n_samples,)
        repetitions: ndarray of shape (n_samples,) or None
        file_path: str
    """
    file_name = f"S{subject}_E{exercise}_A{acquisition}.mat"
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Make sure your Ninapro DB2 file is inside the '{data_dir}' folder."
        )

    data = sio.loadmat(file_path)

    emg = _get_first_existing_key(data, ["emg", "EMG", "data", "X"])
    labels = _get_first_existing_key(data, ["restimulus", "stimulus", "label", "labels", "y"])
    repetitions = _get_first_existing_key(data, ["rerepetition", "repetition", "rep"])

    if emg is None:
        raise KeyError("Could not find EMG array in the .mat file.")
    if labels is None:
        raise KeyError("Could not find label/restimulus array in the .mat file.")

    emg = np.asarray(emg, dtype=np.float64)
    labels = np.asarray(labels).squeeze()

    if repetitions is not None:
        repetitions = np.asarray(repetitions).squeeze()

    if emg.ndim != 2:
        raise ValueError(f"Expected EMG to be 2D, got shape {emg.shape}")

    return emg, labels, repetitions, file_path