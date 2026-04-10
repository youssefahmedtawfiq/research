# user_data_proc.py
import numpy as np
# استدعاء دالة الفلترة من ملفاتك الأصلية
from signalRead_Procescing.preprocessing import preprocess_emg
# نفس الرقم اللي استخدمناه في التدريب لفك التشفير (Decoding) للخرج فقط
MAX_EXPECTED_SPIKES = 50.0 

def process_and_encode_window(window_data, fs=1000, window_ms=200):
    """
    تستقبل نافذة زمنية كاملة، تقوم بفلترتها، ثم عمل Normalization لها،
    وبعد ذلك تشفيرها لنبضات (Spikes) للدخل.
    """
    # ==========================================
    # 1. الفلترة (Filtration & Preprocessing)
    # ==========================================
    try:
        filtered_window = preprocess_emg(
            window_data,
            fs=fs,
            highpass_hz=20.0,
            notch_hz=50.0,
            lowpass_hz=450.0,
            filter_order=4
        )
    except Exception as e:
        print(f"⚠️ Filter warning: {e}")
        filtered_window = window_data

    # ==========================================
    # 2. التسوية (Normalization) 
    # ==========================================
    # أخذ القيمة المطلقة للإشارة المفلترة (Rectification)
    rectified_window = np.abs(filtered_window)
    
    # تطبيق Min-Max Normalization لجعل القيم بين 0.0 و 1.0
    min_val = np.min(rectified_window, axis=0)
    max_val = np.max(rectified_window, axis=0)
    window_01 = (rectified_window - min_val) / (max_val - min_val + 1e-8)

    # ==========================================
    # 3. التشفير (Encoding) للطبقة الأولى (Input)
    # ==========================================
    indices = []
    times = []
    
    # حساب متوسط الإشارة (التي أصبحت الآن بين 0 و 1) لكل حساس
    mean_signals = np.mean(window_01, axis=0)
    
    for neuron_idx, signal_strength in enumerate(mean_signals):
        # هنا نضرب القيمة (0 إلى 1) في رقم كبير لتوليد نبضات الدخل 
        # (مثلاً 100 أو نفس الرقم الذي استخدمته في دالة rate_encoding في التدريب)
        # هذا الرقم يخص مولد النبضات ولا علاقة له بـ MAX_EXPECTED_SPIKES الخاص بالخرج
        num_spikes = int(signal_strength * 100) 
        
        if num_spikes > 0:
            # 1. توزيع النبضات عشوائياً
            raw_spike_times = np.random.uniform(1, window_ms - 1, num_spikes)
            
            # 2. 🌟 الحل: التقريب لأقرب dt (0.1) وإزالة أي أزمنة متكررة في نفس اللحظة 🌟
            unique_spike_times = np.unique(np.round(raw_spike_times, decimals=1))
            
            # 3. تحديث العدد الفعلي بعد إزالة المتكرر
            actual_num_spikes = len(unique_spike_times)
            
            indices.extend([neuron_idx] * actual_num_spikes)
            times.extend(unique_spike_times)
            
    return np.array(indices), np.array(times)


def decode_spikes_to_angles(output_spikes_count):
    """
    تحويل النبضات الخارجة من الأعصاب الحقيقية إلى زوايا
    هنا نستخدم MAX_EXPECTED_SPIKES (20) لأن الأعصاب لها حدود فسيولوجية
    """
    decoded_angles = np.clip(output_spikes_count / MAX_EXPECTED_SPIKES, 0.0, 1.0)
    return decoded_angles