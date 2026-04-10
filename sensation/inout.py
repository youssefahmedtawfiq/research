# core/inout.py
import numpy as np
import os

# 🌟 إعدادات شبكة التغذية الراجعة (Feedback Network Settings)
INPUT_NEURONS = 12 
OUTPUT_NEURONS = 3 # (Volt, Current, Freq)
NEURON_STATE = 1  

# 1. تحديث مسارات ملفات الداتا الجديدة
# تأكد أن هذه الملفات تم إنتاجها من كود التجهيز (feedback_processing.py)
emg_file_path = "processed_data/feedback_input_spikes.npz" 
angles_file_path = "processed_data/feedback_target_signals.npz"

def load_network_data():
    """
    دالة لقراءة الملفات وسحب مصفوفات الحرارة والتحفيز الكهربائي.
    """
    try:
        if not os.path.exists(emg_file_path) or not os.path.exists(angles_file_path):
            print(f"⚠️ خطأ: الملفات غير موجودة في {emg_file_path}")
            return None, None

        # 2. قراءة الملفات
        emg_loaded = np.load(emg_file_path)
        angles_loaded = np.load(angles_file_path)

        # 3. سحب المصفوفات بالأسماء الجديدة (Keys)
        # تم تغيير 'rate_spikes' و 'targets' لتناسب ملفات الـ Feedback
        emg_array = emg_loaded['rate_spikes'] 
        angles_array = angles_loaded['targets'] # تم تعديلها من glove_normalized_data لـ targets

        print(f"✅ Feedback NPZ read successfully. (Input: {emg_array.shape}, Target: {angles_array.shape})")
        
        return emg_array, angles_array

    except Exception as e:
        print(f"❌ Error during loading data: {e}")
        return None, None

# سحب البيانات عند الاستدعاء
emg_var, angles_var = load_network_data()

# ------------------------------------------------------------
# الدوال الخاصة بالأوزان (تغيير اسم الملف لتمييزه عن شبكة الحركة)
# ------------------------------------------------------------

def load_weights(filepath="processed_data/feedback_network_weights.npz"):
    """تحميل أوزان شبكة الـ Feedback"""
    if os.path.exists(filepath):
        try:
            data = np.load(filepath)
            print(">> تم العثور على أوزان سابقة لشبكة الـ Feedback...")
            return {'w1': data['w_in_hid'], 'w2': data['w_hid_out']}
        except Exception as e:
            print(f"حدث خطأ أثناء قراءة الأوزان: {e}")
            return None
    else:
        print(">> لا توجد أوزان سابقة (شبكة Feedback جديدة).")
        return None

def save_weights(w_in_hid, w_hid_out, filepath="processed_data/feedback_network_weights.npz"):
    """حفظ أوزان شبكة الـ Feedback"""
    np.savez_compressed(filepath, w_in_hid=np.array(w_in_hid), w_hid_out=np.array(w_hid_out))
    print(f">> تم حفظ أوزان الـ Feedback بنجاح في: {filepath}")

# ------------------------------------------------------------
# حفظ بيانات الـ Online Feedback
# ------------------------------------------------------------

def append_online_feedback_data(temp_input, predicted_stimulus, filename="processed_data/user_feedback_history.npz"):
    """
    حفظ قراءات الحرارة التي تعرض لها المستخدم والنبضات الكهربائية التي ولدت في ملف منفصل.
    """
    if os.path.exists(filename):
        data = np.load(filename)
        temp_history = data['temp']
        stim_history = data['stimulus']
        
        new_temp = np.expand_dims(temp_input, axis=0)
        new_stim = np.expand_dims(predicted_stimulus, axis=0)
        
        temp_history = np.append(temp_history, new_temp, axis=0)
        stim_history = np.append(stim_history, new_stim, axis=0)
    else:
        temp_history = np.array([temp_input])
        stim_history = np.array([predicted_stimulus])
        
    np.savez_compressed(filename, temp=temp_history, stimulus=stim_history)
    print(f"📁 [Feedback DB] Saved to {filename}. Total samples: {len(temp_history)}")