# main.py
from random import seed

import inout
from train.train import train_network
import numpy as np

def main():
    print("="*50)
    print("1. جاري سحب البيانات من ملفات الـ NPZ...")
    print("="*50)
    
    # 1. سحب البيانات عبر ملف inout (يجب أن تكون دالة load_network_data جاهزة فيه كما اتفقنا)
    emg_data, angles_data = inout.load_network_data()
    #print("\n--- 🔍 فحص مصفوفة الزوايا (Target Data Check) 🔍 ---")
    #changes = 0
    # هنفحص أول 1000 عينة عشان نشوف الزاوية بتتغير ولا ثابتة
    #for i in range(1, min(1000, len(angles_data))):
        # لو الزاوية الحالية مش متطابقة مع الزاوية اللي قبلها
     #   if not np.allclose(angles_data[i], angles_data[i-1], atol=1e-4):
      #      changes += 1
       #     if changes <= 5: # نطبع أول 5 تغييرات بس
        #        print(f"✅ الزاوية اتغيرت عند العينة رقم: {i}")
    
 #   print(f"\nإجمالي عدد التغييرات في أول 1000 عينة = {changes}")
  #  print("--------------------------------------------------\n")
    
    # وقف الكود هنا مؤقتاً عشان منضيعش وقت في التدريب
   # return
    if emg_data is None or angles_data is None:
        print("خطأ: لم يتم العثور على البيانات. يرجى مراجعة المسارات في inout.py")
        return

    print(f"تم سحب البيانات بنجاح: EMG shape = {emg_data.shape}, Angles shape = {angles_data.shape}")

    print("\n" + "="*50)
    print("2. بدء عملية التدريب (Offline Learning)...")
    print("="*50)
    
    print("\n🏆 --- بدء التدريب لجميع الـ 5 States --- 🏆")
    
    # قواميس لحفظ داتا الـ 5 حالات
    all_final_weights = {}
    all_histories = {}
    
    for state in range(1, 2):
        seed(42)
        print(f"\n" + "="*50)
        print(f"🚀 جاري التدريب على State رقم: {state}")
        inout.NEURON_STATE = state
        
        # تشغيل التدريب
        net, inp, S_out, S_in, spk_mon, h_m_in, h_m_out, h_all_in, h_all_out, final_acc = train_network(emg_data, angles_data)
        
        # حفظ الأوزان للـ State الحالي
        all_final_weights[f'w_in_{state}'] = np.array(S_in.w)
        all_final_weights[f'w_out_{state}'] = np.array(S_out.w)
        
        # حفظ الهيستوري للـ State الحالي
        all_histories[f'acc_{state}'] = final_acc
        all_histories[f'h_mean_in_{state}'] = h_m_in
        all_histories[f'h_mean_out_{state}'] = h_m_out
        all_histories[f'h_all_in_{state}'] = h_all_in
        all_histories[f'h_all_out_{state}'] = h_all_out

    # حفظ كل الأوزان في ملف واحد
    np.savez("processed_data/saved_network_weights.npz", **all_final_weights)
    print("✅ تم حفظ أوزان الـ 5 States في ملف saved_network_weights.npz")
    
    # حفظ كل الهيستوري في ملف واحد
    np.savez("processed_data/training_outputs.npz", **all_histories)
    print("✅ تم حفظ تاريخ التدريب للـ 5 States في ملف training_outputs.npz")
    
   # حفظ كل الأوزان في ملف واحد
    np.savez("processed_data/saved_network_weights.npz", **all_final_weights)
    print("✅ تم حفظ أوزان الـ 5 States في ملف saved_network_weights.npz")
    
    # حفظ كل الهيستوري في ملف واحد
    np.savez("processed_data/training_outputs.npz", **all_histories)
    print("✅ تم حفظ تاريخ التدريب للـ 5 States في ملف training_outputs.npz")

if __name__ == "__main__":
    main()