import os
import sys
import numpy as np
import scipy.io as sio

# 🛑 السحر هنا: إجبار بايثون إنه يقرأ الفولدر الحالي قبل ما يعمل أي استدعاء 🛑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ✅ دلوقتي بايثون هيقدر يشوف السطور بتاعتك دي ويقرا الملفات اللي جنبها
from preprocessing import sliding_window_zscore
from encoding import threshold_based_encoding, minmax_normalize_per_feature, rate_encoding

# كمل كودك عادي جداً من هنا...

def main():
    print("=" * 60)
    print("🚀 بدء تجهيز بيانات التغذية الراجعة (Thermal to Electrotactile) ...")
    print("=" * 60)

    # إعدادات التردد والوقت (متوافقة مع البيانات التي ولدناها)
    fs = 2000  
    window_ms = 200
    step_ms = 50
    normalization_window_ms = 200
    rate_time_steps = 20

    output_dir = "../processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 1. سحب البيانات من ملفات الماتلاب (الـ Dataset الجديدة)
    # -----------------------------
    try:
        # تحميل 12 قناة حرارة
        temp_mat = sio.loadmat('data/thermal_input_12ch.mat')
        temp_raw = temp_mat['temp'] 
        
        # تحميل 3 إشارات تحكم (Volt, Current, Freq)
        signals_mat = sio.loadmat('data/electrotactile_output.mat')
        stimulus_raw = signals_mat['signals']
        
        print("✅ Loaded Thermal Input (12 channels):", temp_raw.shape)
        print("✅ Loaded Stimulus Target (3 channels):", stimulus_raw.shape)
    except FileNotFoundError as e:
        print(f"❌ Error: تأكد من وجود ملفات الماتلاب في نفس المجلد: {e}")
        return

    # -----------------------------
    # 2. Preprocessing (Normalization)
    # -----------------------------
    # ملحوظة: لم نستخدم Filter لأن بيانات الحرارة لا تحتاج High-pass (ترددها منخفض)
    normalization_window_samples = int((normalization_window_ms / 1000.0) * fs)
    
    # عمل Z-score لتوحيد نطاق الحرارة عبر الـ 12 قناة
    temp_normalized = sliding_window_zscore(
        temp_raw,
        window_size_samples=normalization_window_samples
    )

    # Normalization للأهداف (0-1) لسهولة تدريب الـ SNN
    min_val = np.min(stimulus_raw, axis=0)
    max_val = np.max(stimulus_raw, axis=0)
    stimulus_normalized = (stimulus_raw - min_val) / (max_val - min_val + 1e-8)

    # -----------------------------
    # 3. Sliding Window Segmentation (التقطيع المتزامن)
    # -----------------------------
    window_size = int((window_ms / 1000.0) * fs)
    step_size = int((step_ms / 1000.0) * fs)

    temp_segments_list = []
    stim_segments_list = []

    # تقطيع المدخلات والمخرجات بشكل متزامن تماماً
    for i in range(0, len(temp_normalized) - window_size + 1, step_size):
        temp_segments_list.append(temp_normalized[i : i + window_size])
        stim_segments_list.append(stimulus_normalized[i : i + window_size])

    segments = np.array(temp_segments_list)
    stim_segments = np.array(stim_segments_list)

    print(f"✅ Created {len(segments)} windows.")
    print("-" * 60)
    
    # -----------------------------
    # 4. Encoding (تحويل الحرارة إلى Spikes)
    # -----------------------------
    # تسوية البيانات بين 0 و 1 قبل التشفير المعدلي (Rate Encoding)
    segments_01 = minmax_normalize_per_feature(segments)
    rate_spikes = rate_encoding(
        segments_01,
        time_steps=rate_time_steps,
        random_seed=42
    )
    
    # -----------------------------
    # 5. حفظ المخرجات النهائية (NPZ)
    # -----------------------------
    # حفظ المدخلات (الحرارة المشفرة)
    input_save_path = os.path.join(output_dir, "feedback_input_spikes.npz")
    np.savez(
        input_save_path,
        segments=segments,      # البيانات الخام المقطعة
        rate_spikes=rate_spikes # النبضات الجاهزة للـ SNN
    )

    # حفظ الأهداف (إشارات الجهاز المطلوبة)
    target_save_path = os.path.join(output_dir, "feedback_target_signals.npz")
    np.savez_compressed(
        target_save_path, 
        targets=stim_segments # القيم الـ 3 (V, I, F) المقطعة
    )

    print(f"📂 Saved Input Spikes to: {input_save_path}")
    print(f"📂 Saved Target Signals to: {target_save_path}")
    print("=" * 60)
    print("🎉 تم التجهيز بنجاح! الشبكة الثانية جاهزة للتدريب الآن.")

if __name__ == "__main__":
    main()