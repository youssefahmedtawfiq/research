# train/train2.py
from brian2 import *
import numpy as np

def online_learning_step(net_objects, user_input_rates, memory_state):
    # فك الكائنات المستلمة من الشبكة
    net, inp_group, S_hid_out, S_in_hid, spike_mon_out = net_objects
    
    alpha_memory = 0.6
    learning_rate = 0.5
    
    prev_spikes = np.copy(np.array(spike_mon_out.count))
    
    # 2. إدخال بيانات المستخدم وتشغيل الشبكة
    inp_group.rates = user_input_rates * Hz
    net.run(250*ms)
    
    # [BLOCK]: استخراج البيانات من الشبكة بعد المعالجة (Processing)
    # هذه البيانات هي نتاج تفاعل الخلايا العصبية مع المدخلات الحالية
    current_spikes = np.array(spike_mon_out.count)
    spikes_delta = current_spikes - prev_spikes
    
    # حساب الثقة والناتج المتوقع
    # حساب الثقة والناتج المتوقع
    if np.sum(spikes_delta) > 0:
        print(f"⚡ Raw Output Spikes: Neuron [0] fired {spikes_delta[0]} spikes | Neuron [1] fired {spikes_delta[1]} spikes")
        winner = np.argmax(spikes_delta)
        sorted_spikes = np.sort(spikes_delta)
        confidence = (sorted_spikes[-1] - sorted_spikes[-2]) / np.sum(spikes_delta) if len(sorted_spikes) > 1 else 1.0
        
        # ==========================================
        # التعديل هنا: لو حصل تعادل في النبضات، مفيش فائز!
        if len(sorted_spikes) > 1 and sorted_spikes[-1] == sorted_spikes[-2]:
            winner = -1 # -1 تعني محتار أو مفيش قرار
            confidence = 0.0
        # ==========================================
        
    else:
        winner = -1
        confidence = 0.0

    print(f"-> Predicted Class: {winner} | Confidence: {confidence*100:.1f}%")

    # قرار التحديث بناءً على الثقة الذاتية (بدون Target خارجي)
    if confidence >= 0.10: 
        print("✅ High Confidence: Self-Reinforcing this prediction...")
        
        # بما أننا لا نملك Target، نعتبر الـ winner هو الـ Target (Pseudo-label)
        # ونعطي مكافأة كاملة لأن الشبكة واثقة في اختيارها
        reward = 1.0 
        
        current_delta = learning_rate * reward * S_hid_out.delta_w
        memory_update = (1 - alpha_memory) * current_delta + alpha_memory * memory_state['last_delta']
        
        S_hid_out.w += memory_update
        S_hid_out.w = np.clip(S_hid_out.w, 0, 15)
        
        memory_state['last_delta'] = memory_update
        
        S_hid_out.delta_w = 0 
        S_in_hid.delta_w = 0
       # طباعة متوسط الأوزان للطبقة الأخيرة (للجراف)
        current_mean_w = np.mean(S_hid_out.w)
        print(f"-> New Mean Weight (Hidden->Out): {current_mean_w:.4f}")
        
        # طباعة مصفوفة الأوزان (Input -> Hidden) - عددهم 16
        print(f"-> Weights (Input -> Hidden):\n{np.array(S_in_hid.w)}")
        
        # طباعة مصفوفة الأوزان (Hidden -> Output) - عددهم 8
        print(f"-> Weights (Hidden -> Output):\n{np.array(S_hid_out.w)}")
        return True, np.mean(S_hid_out.w), confidence, winner
    else:
        print("❌ Low Confidence: Input ignored to keep weights stable.")
        S_hid_out.delta_w = 0 
        S_in_hid.delta_w = 0
        return False, None, confidence, winner