# core/emg_ninapro_db2_subject1.py
import os
import numpy as np
import scipy.io as sio

from dataset import load_ninapro_db2
from preprocessing import resample_emg, preprocess_emg, sliding_window_zscore
from encoding import threshold_based_encoding, minmax_normalize_per_feature, rate_encoding

def main():
    print("=" * 60)
    print("🚀 بدء تجهيز البيانات (EMG & Kinematics) ...")
    print("=" * 60)

    subject = 1
    exercise = 2
    acquisition = 1
    data_dir = "data"

    raw_fs = 2000          
    target_fs = 1000       
    do_resample = True

    window_ms = 200
    step_ms = 50
    normalization_window_ms = 200

    binary_threshold_mode = "median"
    rate_time_steps = 20

    output_dir = "../processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. سحب بيانات الـ EMG الأصلية
    emg_raw, labels, repetitions, file_path = load_ninapro_db2(
        subject=subject,
        exercise=exercise,
        acquisition=acquisition,
        data_dir=data_dir,
    )
    
    # 2. سحب بيانات الزوايا (Glove) من نفس الملف
    mat_data = sio.loadmat(file_path)
    glove_raw = mat_data['glove']

    print("Loaded file:", file_path)
    print("Raw EMG shape:", emg_raw.shape)
    print("Raw Glove shape:", glove_raw.shape)
    print("-" * 60)

    # -----------------------------
    # Resampling (للاتنين مع بعض)
    # -----------------------------
    fs = raw_fs
    if do_resample and raw_fs != target_fs:
        emg = resample_emg(emg_raw, orig_fs=raw_fs, target_fs=target_fs)
        
        # استخراج نفس الـ indices اللي استخدمناها للـ EMG عشان نطبقها على الزوايا
        orig_len = len(labels)
        new_len = emg.shape[0]
        mapped_idx = np.round(np.linspace(0, orig_len - 1, new_len)).astype(int)
        
        labels = labels[mapped_idx]
        if repetitions is not None:
            repetitions = repetitions[mapped_idx]
            
        # تطبيق الـ Resampling على الزوايا بنفس الـ Index!
        glove_resampled = glove_raw[mapped_idx]
        fs = target_fs
    else:
        emg = emg_raw.copy()
        glove_resampled = glove_raw.copy()

    print(f"Sampling frequency used: {fs} Hz")
    print("EMG shape after resampling:", emg.shape)
    print("Glove shape after resampling:", glove_resampled.shape)
    print("-" * 60)

    # -----------------------------
    # Preprocessing (EMG & Glove)
    # -----------------------------
    # فلترة الـ EMG
    emg_filtered = preprocess_emg(
        emg,
        fs=fs,
        highpass_hz=20.0,
        notch_hz=50.0,
        lowpass_hz=450.0,
        filter_order=4,
    )
    normalization_window_samples = int((normalization_window_ms / 1000.0) * fs)
    emg_normalized = sliding_window_zscore(
        emg_filtered,
        window_size_samples=normalization_window_samples
    )

    # Normalization للزوايا (Min-Max)
    min_val = np.min(glove_resampled, axis=0)
    max_val = np.max(glove_resampled, axis=0)
    glove_normalized = (glove_resampled - min_val) / (max_val - min_val + 1e-8)

    # -----------------------------
    # Sliding Window Segmentation (التقطيع المتزامن)
    # -----------------------------
    window_size = int((window_ms / 1000.0) * fs)
    step_size = int((step_ms / 1000.0) * fs)

    emg_segments_list = []
    glove_segments_list = []
    labels_list = []

    # هنمشي على الداتا الموازية ونقطعهم هما الاتنين مع بعض بالمللي!
    for i in range(0, len(emg_normalized) - window_size + 1, step_size):
        emg_segments_list.append(emg_normalized[i : i + window_size])
        glove_segments_list.append(glove_normalized[i : i + window_size])
        labels_list.append(labels[i + window_size - 1])

    segments = np.array(emg_segments_list)
    glove_segments = np.array(glove_segments_list)
    window_labels = np.array(labels_list)

    print("✅ Segments shape (EMG Inputs):", segments.shape)
    print("✅ Segments shape (Glove Targets):", glove_segments.shape)
    print("-" * 60)
    
    # -----------------------------
    # Encoding (لـ EMG فقط)
    # -----------------------------
    segments_01 = minmax_normalize_per_feature(segments)
    rate_spikes = rate_encoding(
        segments_01,
        time_steps=rate_time_steps,
        random_seed=42
    )
    
    # -----------------------------
    # Save Outputs
    # -----------------------------
    # 1. حفظ ملف الـ EMG
    emg_save_path = os.path.join(output_dir, f"S{subject}_E{exercise}_spike_train.npz")
    np.savez(
        emg_save_path,
        segments=segments,       
        rate_spikes=rate_spikes, 
        labels=window_labels
    )
    print(f"Saved EMG pipeline outputs to: {emg_save_path}")

    # 2. حفظ ملف الزوايا (متقطعة وجاهزة)
    glove_save_path = os.path.join(output_dir, "normalized_glove.npz")
    np.savez_compressed(
        glove_save_path, 
        glove_normalized_data=glove_segments # ركز هنا: بنحفظ الـ segments مش الداتا الخام
    )
    print(f"Saved Glove target outputs to: {glove_save_path}")
    print("=" * 60)
    print("🎉 تم التجهيز بنجاح! داتا الـ Input والـ Target متطابقين 100%")

if __name__ == "__main__":
    main()