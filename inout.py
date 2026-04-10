
# core/inout.py
import numpy as np
import os

# إعدادات الشبكة
INPUT_NEURONS = 12 # عدله حسب عدد الـ Features في الـ Pipeline
OUTPUT_NEURONS = 22
NEURON_STATE = 1  

# 1. تحديد مسارات ملفات الداتا (قم بتعديلها حسب مسار الملفات على جهازك)
emg_file_path = "processed_data/S1_E2_spike_train.npz"
angles_file_path = "processed_data/normalized_glove.npz"

def load_network_data():
    """
    دالة لقراءة الملفات وسحب المصفوفات. 
    استخدام دالة أفضل من كتابة الكود بشكل حر لتجنب استهلاك الميموري 
    إلا وقت استدعاء البيانات فعلياً.
    """
    try:
        # 2. قراءة الملفات
        emg_loaded = np.load(emg_file_path)
        angles_loaded = np.load(angles_file_path)

        # 3. سحب الـ Arrays وتخزينها في متغيرات
        # (تأكد أن 'emg_encoded_data' هو نفس الاسم الذي حفظت به ملف الـ EMG)
        emg_array = emg_loaded['rate_spikes'] # أو 'rate_spikes' حسب اختيارك
        angles_array = angles_loaded['glove_normalized_data']

        print("npz. read successfully.")
        
        return emg_array, angles_array

    except Exception as e:
        print(f"error {e}")
        return None, None

# إذا أردت سحب البيانات مباشرة بمجرد عمل import للملف (كما طلبت):
emg_var, angles_var = load_network_data()



# ... (الكود القديم بتاعك زي ما هو) ...

# ضيف الدالتين دول في نهاية ملف inout.py

def load_weights(filepath="processed_data/saved_network_weights.npz"):
    """دالة لتحميل الأوزان القديمة لو موجودة"""
    if os.path.exists(filepath):
        try:
            data = np.load(filepath)
            print(">> تم العثور على أوزان سابقة، جاري تحميلها...")
            return {'w1': data['w_in_hid'], 'w2': data['w_hid_out']}
        except Exception as e:
            print(f"حدث خطأ أثناء قراءة الأوزان: {e}")
            return None
    else:
        print(">> لا توجد أوزان سابقة (ملف جديد)، سيتم التدريب من الصفر.")
        return None

def save_weights(w_in_hid, w_hid_out, filepath="processed_data/saved_network_weights.npz"):
    """دالة لحفظ الأوزان بعد التدريب"""
    np.savez_compressed(filepath, w_in_hid=np.array(w_in_hid), w_hid_out=np.array(w_hid_out))
    print(f">> تم حفظ الأوزان بنجاح في: {filepath}")




def append_online_data(analog_input, predicted_angles, filename="processed_data/user_online_data.npz"):
    """
    حفظ الإشارة التي أدخلها المستخدم والزوايا التي توقعتها الشبكة
    في ملف منفصل لتكوين Dataset خاص بالمستخدم.
    """
    if os.path.exists(filename):
        data = np.load(filename)
        emg_history = data['emg']
        angles_history = data['angles']
        
        new_emg = np.expand_dims(analog_input, axis=0)
        new_angles = np.expand_dims(predicted_angles, axis=0)
        
        emg_history = np.append(emg_history, new_emg, axis=0)
        angles_history = np.append(angles_history, new_angles, axis=0)
    else:
        emg_history = np.array([analog_input])
        angles_history = np.array([predicted_angles])
        
    np.savez_compressed(filename, emg=emg_history, angles=angles_history)
    print(f"📁 [Database] Saved to {filename}. Total user samples: {len(emg_history)}")