# main.py
from brian2 import prefs
import numpy as np
import matplotlib.pyplot as plt

# استدعاء التدريب القديم والجديد
from train.train import train_network
from train.train2 import online_learning_step

prefs.codegen.target = 'numpy'

if __name__ == '__main__':
    print("========================================")
    print(" Spiking Neural Network (Izhikevich)    ")
    print("========================================")
    
    # 1. تشغيل التدريب الأساسي أولاً (Offline)
    print("\n--- Starting Offline Training Phase ---")
    net, inp_group, S_hid_out, S_in_hid, spike_mon_out, weight_history = train_network()
    net_objects = (net, inp_group, S_hid_out, S_in_hid, spike_mon_out)

    # =========================================================
    # ⚠️ ملحوظة هامة جداً:
    # لو ظهرلك الجراف الأول بتاع الـ Offline، الكود هيقف لحد ما تقفله.
    # اقفل نافذة الجراف الأول عشان البرنامج يكمل ويدخل في الـ Online!
    # =========================================================

    # 2. إعداد الجراف المباشر (Interactive Mode) للمرحلة الثانية
    plt.ion() # تشغيل الوضع التفاعلي عشان الكود مايقفش
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(weight_history, marker='o', color='b', label='Weight Evolution')
    plt.title('Real-time Online Learning Weights')
    plt.grid(True)
    plt.show() # هيظهر الجراف ويسيب البرنامج شغال في الخلفية

    # 3. حلقة الـ Online Learning التفاعلية مع المستخدم
    print("\n========================================")
    print(" --- Online Phase Started (User Input) ---")
    print("========================================")
    
    # تهيئة الذاكرة بصفر لتبدأ بناء التحديثات (Memory Trace)
    memory_state = {'last_delta': 0}

    while True:
        # هنا هيسألك وتدخل الأرقام
        user_input = input("\nEnter 4 input rates and 1 target (e.g. 200 0 150 50 0) or 'exit': ")
        
        if user_input.lower() == 'exit': 
            break
        
        try:
            data = [float(x) for x in user_input.split()]
            
            # التأكد إنك دخلت 5 أرقام بالضبط
            if len(data) != 5:
                print("⚠️ خطأ: الرجاء إدخال 5 أرقام (4 للـ input و 1 للـ target) بينهم مسافات.")
                continue
                
            rates = np.array(data[:4])
            target = int(data[4])
            
            # استدعاء خطوة التعلم المباشر من ملف train2
            updated, new_mean_w, correct = online_learning_step(net_objects, rates, target, memory_state)
            
            # لو الثقة كانت كويسة والشبكة حدثت الأوزان:
            if updated:
                weight_history.append(new_mean_w)
                
                # تحديث الجراف فوراً قدام عينك
                line.set_ydata(weight_history)
                line.set_xdata(range(len(weight_history)))
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                print(f"✅ Graph Updated! New Mean Weight: {new_mean_w:.4f}")
                
        except Exception as e:
            print(f"⚠️ Error: {e}. Please check your input format.")

    # لما تكتب exit ويقفل الحلقه
    plt.ioff()
    print("\n✅ Training Session Ended. You can now close the final graph.")
    plt.show()