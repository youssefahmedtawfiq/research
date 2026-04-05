# train/train2.py
from brian2 import *
import numpy as np

def online_learning_step(net_objects, user_input_rates, target, memory_state):
    # فك الكائنات المستلمة من الشبكة
    net, inp_group, S_hid_out, S_in_hid, spike_mon_out = net_objects
    
    # إعدادات الذاكرة والثقة
    alpha_memory = 0.6  # مدى الاعتماد على الذاكرة السابقة (0.0 لـ 1.0)
    learning_rate = 0.5
    
    # 1. تسجيل الحالة قبل المحاكاة
    prev_spikes = np.copy(np.array(spike_mon_out.count))
    
    # 2. إدخال بيانات المستخدم وتشغيل الشبكة
    inp_group.rates = user_input_rates * Hz
    net.run(250*ms)
    
    # 3. حساب النبضات الجديدة (Spikes)
    current_spikes = np.array(spike_mon_out.count)
    spikes_delta = current_spikes - prev_spikes
    
    # 4. اختبار الثقة (Confidence Test)
    # الثقة تعتمد على الفرق بين أعلى نيورون نبض والنيورون الذي يليه
    if np.sum(spikes_delta) > 0:
        winner = np.argmax(spikes_delta)
        sorted_spikes = np.sort(spikes_delta)
        # معادلة الثقة: الفرق بين المركز الأول والثاني مقسوماً على الإجمالي
        confidence = (sorted_spikes[-1] - sorted_spikes[-2]) / np.sum(spikes_delta) if len(sorted_spikes) > 1 else 1.0
    else:
        winner = -1
        confidence = 0.0

    print(f"-> Prediction: {winner} | Confidence: {confidence*100:.1f}%")

    # 5. قرار التحديث (لو الثقة كويسة يعدل)
    if confidence >= 0.05: # حد الثقة (يمكنك تعديله)
        print("✅ Confidence High: Updating Weights with Memory Dependency...")
        reward = 1.0 if winner == target else -0.8
        
        # معادلة الذاكرة: التحديث الجديد يتأثر بما قبله
        # New_Update = (1-alpha)*Current_Delta + alpha*Previous_Delta
        current_delta = learning_rate * reward * S_hid_out.delta_w
        memory_update = (1 - alpha_memory) * current_delta + alpha_memory * memory_state['last_delta']
        
        # تحديث الأوزان فعلياً
        S_hid_out.w += memory_update
        S_hid_out.w = np.clip(S_hid_out.w, 0, 15)
        
        # حفظ "الذاكرة" للخطوة القادمة
        memory_state['last_delta'] = memory_update
        
        # تصفير الأثر (Eligibility Trace)
        S_hid_out.delta_w = 0 
        S_in_hid.delta_w = 0
        
        return True, np.mean(S_hid_out.w), winner == target
    else:
        print("❌ Confidence Low: Network ignored this input to protect stability.")
        S_hid_out.delta_w = 0 
        S_in_hid.delta_w = 0
        return False, None, False