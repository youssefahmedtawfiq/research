import os
import sys
import numpy as np

# إجبار بايثون على رؤية المجلد الحالي والمجلد الرئيسي
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from preprocessing import sliding_window_zscore
    from encoding import minmax_normalize_per_feature, rate_encoding
except ImportError:
    print("⚠️ تأكد من وجود ملفات preprocessing.py و encoding.py في نفس هذا المجلد")

def process_and_encode_window(analog_window):
    """
    الدالة التي يستدعيها ملف online_learning لتحويل الحرارة لنبضات
    Input: analog_window (200, 12)
    """
    # تحويل البيانات لتنسيق (Batch, Time, Channels)
    segment = analog_window.reshape(1, 200, 12)
    
    # تسوية البيانات (0 to 1)
    segment_01 = minmax_normalize_per_feature(segment)
    
    # التشفير المعدلي (20 خطوة زمنية)
    rate_spikes = rate_encoding(segment_01, time_steps=20)
    
    # استخراج مواضع النبضات (Indices, Times) لـ Brian2
    indices, times = np.where(rate_spikes[0] > 0)
    
    return indices, times

def main():
    print("🚀 Sensation Processing Module Loaded Successfully.")

if __name__ == "__main__":
    main()