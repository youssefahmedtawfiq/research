# core/plots.py
import matplotlib.pyplot as plt
import numpy as np


def plot_raw_vs_filtered(raw_emg, filtered_emg, channel=0, num_samples=3000):
    plt.figure(figsize=(12, 5))
    plt.plot(raw_emg[:num_samples, channel], label="Raw EMG")
    plt.plot(filtered_emg[:num_samples, channel], label="Filtered EMG")
    plt.title(f"Raw vs Filtered EMG - Channel {channel}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized_emg(normalized_emg, channel=0, num_samples=3000):
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_emg[:num_samples, channel])
    plt.title(f"Sliding-Window Normalized EMG - Channel {channel}")
    plt.xlabel("Samples")
    plt.ylabel("Normalized amplitude")
    plt.tight_layout()
    plt.show()


def plot_segment(segment, channel=0):
    plt.figure(figsize=(10, 4))
    plt.plot(segment[:, channel])
    plt.title(f"One EMG Segment - Channel {channel}")
    plt.xlabel("Samples in window")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_feature_vector(features_3d, window_index=0, channel=0):
    names = ["MAV", "WL", "VAR", "ZC", "SSC"]
    vals = features_3d[window_index, channel]

    plt.figure(figsize=(8, 4))
    plt.bar(names, vals)
    plt.title(f"Features for window {window_index}, channel {channel}")
    plt.tight_layout()
    plt.show()


def plot_binary_spikes(binary_spikes, num_windows=50, num_features=30):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        binary_spikes[:num_windows, :num_features].T,
        aspect="auto",
        interpolation="nearest"
    )
    plt.title("Binary Spike Encoding")
    plt.xlabel("Window index")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.show()


def plot_rate_spikes(rate_spikes, window_index=0, num_features=30):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        rate_spikes[window_index, :, :num_features].T,
        aspect="auto",
        interpolation="nearest"
    )
    plt.title(f"Rate Encoding - Window {window_index}")
    plt.xlabel("Time step")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.show()