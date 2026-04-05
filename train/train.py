# train/train.py
from brian2 import *
from model.network import create_network
import inout
import matplotlib.pyplot as plt
import numpy as np

def train_network():
    # استدعاء الشبكة
    inp_group, hidden_group, output_group, S_in_hid, S_hid_out = create_network()
    
    # بناء الـ Network object في Brian2
    net = Network(inp_group, hidden_group, output_group, S_in_hid, S_hid_out)
    
    # مراقب الـ Spikes لمعرفة أي Output Neuron فاز
    spike_mon_out = SpikeMonitor(output_group)
    net.add(spike_mon_out)

    print("\n--- Initial Weights (Hidden -> Output) ---")
    initial_w = np.copy(S_hid_out.w)
    print(initial_w)

    epochs = 5
    learning_rate = 0.5
    weight_history = [np.mean(initial_w)] # لتسجيل متوسط الوزن للرسم
    
    correct_predictions = 0
    total_samples = epochs * len(inout.train_data)
    prev_spikes = np.zeros(inout.OUTPUT_NEURONS)

    print("\n--- Starting RSTDP Training ---")
    for epoch in range(epochs):
        for idx, data in enumerate(inout.train_data):
            # 1. إدخال البيانات
            inp_group.rates = data['input'] * Hz
            target = data['target']
            
            # 2. تشغيل المحاكاة لهذا النمط لمدة 100 ملي ثانية
            net.run(100*ms)
            
            # 3. حساب من الفائز
            current_total_spikes = np.array(spike_mon_out.count)
            spikes_in_this_run = current_total_spikes - prev_spikes
            prev_spikes = np.copy(current_total_spikes)
            
            predicted = np.argmax(spikes_in_this_run) if np.sum(spikes_in_this_run) > 0 else -1
            
            # 4. نظام المكافأة والعقاب (RSTDP Logic)
            if predicted == target:
                reward = 1.0
                correct_predictions += 1
            else:
                reward = -0.5 # عقاب بسيط إذا أخطأ
                
            # 5. تحديث الأوزان وطباعة الجديد
            S_hid_out.w += learning_rate * reward * S_hid_out.delta_w
            S_in_hid.w += learning_rate * reward * S_in_hid.delta_w
            
            # تصفير الـ Eligibility Trace للنمط القادم
            S_hid_out.delta_w = 0 
            S_in_hid.delta_w = 0
            
            # منع الأوزان من أن تصبح سالبة جداً أو عملاقة
            S_hid_out.w = np.clip(S_hid_out.w, 0, 15)
            S_in_hid.w = np.clip(S_in_hid.w, 0, 15)
            
            current_mean_weight = np.mean(S_hid_out.w)
            weight_history.append(current_mean_weight)
            print(f"Epoch {epoch+1}, Sample {idx+1} | Target: {target}, Pred: {predicted} | Reward: {reward} | New Mean W: {current_mean_weight:.4f}")

    print("\n--- Final Weights (Hidden -> Output) ---")
    print(np.array(S_hid_out.w))
    
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\nTraining Completed. Overall Accuracy: {accuracy:.2f}%")

    # 6. رسم الـ Graph
    plt.figure(figsize=(10, 5))
    plt.plot(weight_history, marker='o', linestyle='-', color='b')
    plt.title('Average Hidden-to-Output Weights Evolution (RSTDP)')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Synaptic Weight')
    plt.grid(True)
    plt.show()
    # في نهاية ملف train/train.py أضف هذا السطر:
    return net, inp_group, S_hid_out, S_in_hid, spike_mon_out, weight_history