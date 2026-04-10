# train/train.py
from brian2 import *
from model.network import create_network
import inout
import numpy as np

def train_network(emg_inputs, target_angles):



    # 🌟 إضافة: تعريف أعداد النيورونات للطباعة (مكان السطر 12 تقريباً)
 

    # 1. استدعاء الشبكة (تمت إضافة S_hid_hid لتجنب التحذير)
    inp_group, hidden_group, output_group, S_in_hid, S_hid_out, S_hid_hid = create_network()
    net = Network(inp_group, hidden_group, output_group, S_in_hid, S_hid_out, S_hid_hid)

    #delete
    n_in, n_hid, n_out = len(inp_group), len(hidden_group), len(output_group)
    print(f"\n[Network Info] Total Neurons: {n_in + n_hid + n_out} (In:{n_in}, Hid:{n_hid}, Out:{n_out})")
    #delete



    spike_mon_out = SpikeMonitor(output_group)
    net.add(spike_mon_out)

    # 2. تحميل الأوزان القديمة
    saved_w = inout.load_weights()
    if saved_w is not None:
        S_in_hid.w = saved_w['w1']
        S_hid_out.w = saved_w['w2']

    print("\n--- Initial Weights (Hidden -> Output) ---")
    initial_w = np.copy(S_hid_out.w)
    # طباعة متوسط الأوزان بدلاً من طباعتها كلها لتخفيف الزحام على الشاشة
    print(f"Mean Initial Weight: {np.mean(initial_w):.4f}")

    epochs = 100
    learning_rate = 0.05
    weight_history = [np.mean(initial_w)]
    
    # متغيرات لتتبع الأداء التراكمي (Accumulative Metrics)
    cumulative_error = 0.0
    cumulative_similarity = 0.0
    
    prev_spikes = np.zeros(inout.OUTPUT_NEURONS)
    all_predictions = []

    # [مهم]: أقصى عدد نبضات يمكن أن يخرجه النيورون في 200ms للـ Normalization
    # يمكنك تعديل هذا الرقم بناءً على نشاط شبكتك (مثلاً من 15 إلى 30)
    MAX_EXPECTED_SPIKES = 30.0 

    history_mean_in = []
    history_mean_out = []
    history_all_w_in = []
    history_all_w_out = []

    print("\n--- Starting RSTDP Regression Training ---")
    for epoch in range(epochs):
        for idx, (inp_sample, target_sample) in enumerate(zip(emg_inputs, target_angles)):
            
# =======================================================
            # 🌟 1. استخراج وإدخال النبضات (محدث ومزامن مع Brian2) 🌟
            # =======================================================
            
            # استخراج أماكن النبضات والزمن الخاص بها
            time_steps, neuron_indices = np.where(inp_sample > 0)
            
            # الترتيب الزمني (مهم جداً: لضمان دخول النبضات من الأقدم للأحدث)
            sort_idx = np.argsort(time_steps)
            sorted_indices = neuron_indices[sort_idx]
            sorted_times = time_steps[sort_idx]
            
            # مزامنة الزمن المطلق: إضافة وقت الشبكة الحالي + هامش الأمان dt
            times_absolute = (sorted_times * ms) + net.t + defaultclock.dt
            
            # إدخال النبضات للشبكة
            inp_group.set_spikes(sorted_indices, times_absolute)
            
            # =======================================================
            
            # 2. تجهيز الـ Target (22 زاوية متصلة)
            if target_sample.ndim > 1:
                target = np.mean(target_sample, axis=0)
            else:
                target = target_sample
            
            # 3. تشغيل المحاكاة لمدة 200ms (حجم النافذة الزمنية)
            net.run(200*ms)
            
            # 4. حساب النبضات الخارجة من الـ 22 نيورون
            current_total_spikes = np.array(spike_mon_out.count)
            spikes_in_this_run = current_total_spikes - prev_spikes
            prev_spikes = np.copy(current_total_spikes)
            
            # 5. عملية الـ Decoding (تحويل النبضات لإشارة متصلة Normalized)
            decoded_predicted = np.clip(spikes_in_this_run / MAX_EXPECTED_SPIKES, 0.0, 1.0)
            all_predictions.append(decoded_predicted)
            
            # 6. حساب الخطأ المطلق (MAE) والمكافأة
            current_error = np.mean(np.abs(target - decoded_predicted))
            current_similarity = (1.0 - current_error) * 100.0
            
            # حساب المكافأة (Reward): تقل كلما زاد الخطأ، وممكن تكون سالبة لو الخطأ كبير
            reward = 1.0 - (current_error * 2.0)
                

#delete
# 7. تحديث الأوزان (Supervised STDP)
          # 7. تحديث الأوزان (Supervised STDP المستقل - مع تهدئة التعلم)
            error_vector = target - decoded_predicted 
            
            # 🌟 التعديل الأول: تقليل قوة التحديث لأن الشبكة تعمل الآن بشكل جيد
            update_factor = 0.5  
            
            # 🌟 التعديل الثاني: عامل نسيان خفيف لمنع تشبع الأوزان (Homeostasis)
            decay_rate = 0.9999 
            
            w_out_current = np.array(S_hid_out.w)
            j_indices = np.array(S_hid_out.j)
            delta_w_out = np.array(S_hid_out.delta_w)
            
            # تطبيق عامل النسيان ثم إضافة التحديث الجديد
            w_out_new = (w_out_current * decay_rate) + (learning_rate * error_vector[j_indices] * (np.abs(delta_w_out) + 0.1) * update_factor)
            
            w_in_current = np.array(S_in_hid.w)
            delta_w_in = np.array(S_in_hid.delta_w)
            w_in_new = (w_in_current * decay_rate) + (learning_rate * np.mean(error_vector) * (np.abs(delta_w_in) + 0.1) * update_factor)

            # وضع الحدود لضمان استقرار الشبكة
            S_hid_out.w = np.clip(w_out_new, 0.1, 15)
            S_in_hid.w = np.clip(w_in_new, 0.1, 15)

            # تصفير المتغيرات
            S_hid_out.delta_w = 0 
            S_in_hid.delta_w = 0
            S_hid_out.apre = 0
            S_hid_out.apost = 0
            S_in_hid.apre = 0
            S_in_hid.apost = 0
            current_mean_weight = np.mean(S_hid_out.w)
            weight_history.append(current_mean_weight)
            history_mean_in.append(np.mean(S_in_hid.w))
            history_mean_out.append(np.mean(S_hid_out.w))
            history_all_w_in.append(np.array(S_in_hid.w))
            history_all_w_out.append(np.array(S_hid_out.w))
            # 8. حساب التراكميات (Accumulative)
            current_step = epoch * len(emg_inputs) + idx + 1
            cumulative_error += current_error
            cumulative_similarity += current_similarity
            
            running_avg_error = cumulative_error / current_step
            running_avg_accuracy = cumulative_similarity / current_step
            input_spikes_count = np.sum(inp_sample, axis=0)
            
            # 9. الطباعة الشاملة
        # يستحسن تخليها تطبع كل 100 عينة عشان الشاشة متجريش بسرعة
            print(f"\n[Epoch {epoch+1} | Sample {idx+1}/{len(emg_inputs)}]")
            print(f"Input Spikes (12): {input_spikes_count}") # <--- ده السطر الجديد
            print(f"Target       (22): {np.round(target, 2)}")
            print(f"Pred_Dec     (22): {np.round(decoded_predicted, 2)}")
            print(f"Out Spikes   (22): {spikes_in_this_run}")
            print(f"Step Error: {current_error:.4f}  | Mean W: {current_mean_weight:.4f}")
            # استبدل سطر الطباعة القديم بهذا السطر (أنظف وأدق)
            print(f"Step Error (Avg): {current_error:.4f} | Mean W: {current_mean_weight:.4f} | Acc. Accuracy: {running_avg_accuracy:.2f}%")
            print(f">> Accumlative Error: {running_avg_error:.4f} | Accumlative Accuracy: {running_avg_accuracy:.2f}%")
           
           
           #delete
            # 🌟 إضافة: طباعة متوسط الأوزان لكل نيورون بشكل مستقل
            # بنحسب متوسط الأوزان اللي داخلة لكل نيورون (In) واللي خارجة لكل نيورون (Out)
          # طريقة الطباعة الدقيقة
            mean_w_in_per_neuron = [np.mean(np.array(S_in_hid.w)[np.array(S_in_hid.i) == k]) for k in range(n_in)]
            mean_w_out_per_neuron = [np.mean(np.array(S_hid_out.w)[np.array(S_hid_out.j) == k]) for k in range(n_out)]
            
            print(f"Mean W per In_Neuron (12): {np.round(mean_w_in_per_neuron, 4)}")
            print(f"Mean W per Out_Neuron (22): {np.round(mean_w_out_per_neuron, 4)}")
           
           
           
           
           
           
           
            print("-" * 60)
            
    print("\n--- Final Weights (Hidden -> Output) ---")
    print(f"Mean Final Weight: {np.mean(S_hid_out.w):.4f}")
    
    print(f"\nTraining Completed. Final Accumulative Accuracy: {running_avg_accuracy:.2f}%")
    return net, inp_group, S_hid_out, S_in_hid, spike_mon_out, history_mean_in, history_mean_out, history_all_w_in, history_all_w_out, running_avg_accuracy
