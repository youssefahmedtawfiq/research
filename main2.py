# main.py
from brian2 import prefs
import numpy as np
import matplotlib.pyplot as plt
from train.train import train_network
from train.train2 import online_learning_step

prefs.codegen.target = 'numpy'

if __name__ == '__main__':
    # المرحلة الأولى: التدريب الأساسي
    net, inp_group, S_hid_out, S_in_hid, spike_mon_out, weight_history = train_network()
    net_objects = (net, inp_group, S_hid_out, S_in_hid, spike_mon_out)

    # إعداد جرافين: واحد للأوزان وواحد للدقة (Confidence Rate)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    line_w, = ax1.plot(weight_history, marker='o', color='b', label='Mean Weight')
    ax1.set_title('Synaptic Weights Evolution')
    ax1.grid(True)
    
    accuracy_history = [0.0]
    line_acc, = ax2.plot(accuracy_history, marker='s', color='g', label='Confidence Accuracy')
    ax2.set_title('Online Confidence Success Rate (%)')
    ax2.set_ylim(0, 110)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    print("\n--- Online Learning Phase (Target-Free) ---")
    memory_state = {'last_delta': 0}
    total_inputs = 0
    confident_decisions = 0

  # جزء من ملف main.py

    while True:
        user_input = input("\nEnter 4 input rates (e.g. 300 0 100 0) or 'exit': ")
        
        if user_input.lower() == 'exit': break
        
        try:
            rates = np.array([float(x) for x in user_input.split()])
            if len(rates) != 4:
                print("⚠️ الرجاء إدخال 4 أرقام فقط.")
                continue
                
            total_inputs += 1 # زيادة عداد المحاولات الكلية
            
            # 1. تنفيذ خطوة التعلم المباشر (تعديل الـ Weight)
            updated, new_mean_w, conf, winner = online_learning_step(net_objects, rates, memory_state)
            
            if updated:
                # 2. تحديث عداد القرارات الواثقة
                confident_decisions += 1
                weight_history.append(new_mean_w)
                
                # 3. عمل Push للبيانات في الـ Dataset Array
                new_data_point = {
                    'input': rates.tolist(),
                    'target': int(winner)
                }
                import inout
                inout.train_data.append(new_data_point)
                
                # 4. حساب الدقة بعد الـ Update والـ Push مباشرة
                current_acc = (confident_decisions / total_inputs) * 100
                accuracy_history.append(current_acc)

                # --- طباعة النتائج النهائية للمستخدم ---
                print(f"📥 [Data Pushed] Dataset Size: {len(inout.train_data)}")
                print(f"📈 [Weight Updated] Mean Weight: {new_mean_w:.4f}")
                print(f"🎯 [Current Accuracy] {current_acc:.2f}% (Decisive Actions: {confident_decisions}/{total_inputs})")
                print("-----------------------------------------")
                
            else:
                # في حالة عدم التحديث (ثقة منخفضة) نحسب الدقة برضه عشان نشوفها بتقل ولا لا
                current_acc = (confident_decisions / total_inputs) * 100
                accuracy_history.append(current_acc)
                print(f"⚠️ [Input Ignored] Confidence too low. Current Accuracy: {current_acc:.2f}%")

            # تحديث الجرافات (الأوزان والدقة)
            line_w.set_ydata(weight_history)
            line_w.set_xdata(range(len(weight_history)))
            ax1.relim()
            ax1.autoscale_view()
            
            line_acc.set_ydata(accuracy_history)
            line_acc.set_xdata(range(len(accuracy_history)))
            ax2.relim()
            ax2.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
    plt.ioff()
    plt.show()