# control.py
import numpy as np
import os

def clean_untrained_weights():
    file_path = "processed_data/saved_network_weights.npz"
    
    print("="*60)
    print("🔧 أداة التحكم وتنظيف الأوزان (Weights Controller) 🔧")
    print("="*60)

    # 1. التأكد إن الملف موجود أصلاً
    if not os.path.exists(file_path):
        print("⚠️ ملف الأوزان مش موجود أصلاً في المسار المحدد!")
        return

    try:
        # 2. قراءة الداتا من الملف
        data = dict(np.load(file_path, allow_pickle=True))
        print(f"📂 تم فتح الملف بنجاح. يحتوي حالياً على الأوزان الآتية:\n   {list(data.keys())}\n")
        
        # 3. تحديد الحالات اللي عايزين نمسحها (تقدر تزود أو تنقص براحتك)
        states_to_delete = ['_2', '_3', '_4', '_5', '_6']
        
        clean_data = {}
        deleted_keys = []
        
        # 4. فلترة الأوزان
        for key, value in data.items():
            # لو اسم الوزن بينتهي برقم من الحالات اللي عايزين نمسحها، هنحطه في سلة المهملات
            if any(key.endswith(state) for state in states_to_delete):
                deleted_keys.append(key)
            else:
                # لو وزن مهم (زي State 1)، هنحتفظ بيه
                clean_data[key] = value
                
        # 5. الحفظ والطباعة
        if deleted_keys:
            print(f"🧹 جاري مسح الأوزان الوهمية: {deleted_keys}")
            np.savez_compressed(file_path, **clean_data)
            print(f"✅ تم تنظيف الملف وحفظه بنجاح!")
            print(f"🔒 الأوزان المتبقية المحفوظة: {list(clean_data.keys())}")
        else:
            print("✨ الملف نظيف بالفعل ولا يحتوي على أوزان للحالات المحددة.")
            
    except Exception as e:
        print(f"❌ حدث خطأ أثناء محاولة تنظيف الملف: {e}")

if __name__ == "__main__":
    clean_untrained_weights()