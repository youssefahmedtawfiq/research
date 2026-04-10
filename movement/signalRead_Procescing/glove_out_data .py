
# 1. تحديد المسارات (قم بتغييرها بما يناسب جهازك)
# مسار ملف الداتا الأصلي الخاص بـ NinaPro
input_file_path = "data/S1_E2_A1.mat" 

# مسار الفولدر اللي إنت عامله عشان تحفظ فيه النتيجة
output_folder = "../processed_data"
output_filename = "normalized_glove.npz"

# 2. قراءة الملف وسحب بيانات الـ glove
# دالة loadmat بتحول الملف لـ Dictionary نقدر نسحب منه المتغيرات
mat_data = sio.loadmat(input_file_path)
glove_data = mat_data['glove']

# 3. عمل Normalization
# هنا بنستخدم طريقة Min-Max Normalization عشان نخلي القيم كلها بين 0 و 1
min_val = np.min(glove_data, axis=0)
max_val = np.max(glove_data, axis=0)

# (نضيف قيمة صغيرة جداً 1e-8 لتجنب خطأ القسمة على صفر لو كانت القيم متطابقة)
glove_normalized = (glove_data - min_val) / (max_val - min_val + 1e-8)

# التأكد من أن الفولدر موجود (ولو مش موجود الكود هيكريته)
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, output_filename)

# =====================================================================
# 4. هذا هو الجزء الذي يبعت الـ dataset (بيانات الـ glove) إلى الملف ويحفظها
# استخدمنا savez_compressed لضغط الملف وتقليل مساحته
np.savez_compressed(save_path, glove_normalized_data=glove_normalized)
# =====================================================================

print(f"the data saved at {save_path}")