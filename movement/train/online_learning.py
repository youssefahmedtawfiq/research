from random import seed
from brian2 import *
import numpy as np
import inout
from signalRead_Procescing.user_data_proc import process_and_encode_window, decode_spikes_to_angles
from model.network import create_network

def process_online_sample(net_objects, analog_emg_input, memory_state, step_idx, accumulative_conf):
    alpha_memory = 0.6  
    learning_rate = 0.05 
    
    indices, times = process_and_encode_window(analog_emg_input)
    if len(indices) == 0:
        return False, memory_state, None, accumulative_conf
        
    weights_file = "processed_data/saved_network_weights.npz"
    weights_found = False
    try:
        saved_weights = np.load(weights_file)
        updated_weights = dict(saved_weights) 
        weights_found = True
    except:
        updated_weights = {}

    best_state, best_confidence, best_predicted_angles = 1, -1.0, None
    successful_states, userdata_to_save = [], {}
    
    print(f"\n{'='*50}\n🧠 processing 5 states (Step: {step_idx})\n{'='*50}")
    
    for state in range(1, 6):
        print(f"\n▶️ State {state}:")
        
        # أمر التخطي السحري
        if weights_found and (f'w_in_{state}' not in updated_weights):
            print(f"   └─ ⏭️ Skipped (تخطي): مفيش أوزان.")
            continue
            
        inout.NEURON_STATE = state
        seed(42) 
        
        inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h = create_network()
        out_mon = SpikeMonitor(out_g)
        temp_net = Network(inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h, out_mon)
        
        if weights_found:
            S_i_h.w = updated_weights.get(f'w_in_{state}', S_i_h.w)
            S_h_o.w = updated_weights.get(f'w_out_{state}', S_h_o.w)
            
        old_w_in, old_w_out = np.array(S_i_h.w), np.array(S_h_o.w)
        mean_weight = (np.mean(old_w_in) + np.mean(old_w_out)) / 2.0
        
        inp_g.set_spikes(indices, times * ms)
        temp_net.run(200*ms)
        
        current_spikes = np.array(out_mon.count)
        total_spikes = np.sum(current_spikes)
        predicted_angles = decode_spikes_to_angles(current_spikes)
        
        # حساب الثقة
        if total_spikes > 5 and np.max(predicted_angles) < 0.95:
            confidence = min((total_spikes / (22.0 * 30.0)), 1.0)
        else:
            confidence = 0.0
            
        print(f"   └─ 📊 Accuracy: {confidence*100:.1f}% | Mean Weight: {mean_weight:.4f}")
            
        if confidence > best_confidence:
            best_confidence, best_state, best_predicted_angles = confidence, state, predicted_angles

        # التحديث
        if confidence >= 0.15:
            print(f"   └─ ✅ Passed: جاري تعديل الأوزان...")
            if f'last_delta_out_{state}' not in memory_state:
                memory_state[f'last_delta_out_{state}'] = 0
                memory_state[f'last_delta_in_{state}'] = 0
                
            memory_update_out = (1 - alpha_memory) * (learning_rate * np.abs(np.array(S_h_o.delta_w)) * confidence) + (alpha_memory * memory_state[f'last_delta_out_{state}'])
            memory_update_in = (1 - alpha_memory) * (learning_rate * np.abs(np.array(S_i_h.delta_w)) * confidence) + (alpha_memory * memory_state[f'last_delta_in_{state}'])
            
            updated_weights[f'w_in_{state}'] = np.clip(old_w_in + memory_update_in, 0.1, 15)
            updated_weights[f'w_out_{state}'] = np.clip(old_w_out + memory_update_out, 0.1, 15)
            
            memory_state[f'last_delta_out_{state}'] = memory_update_out
            memory_state[f'last_delta_in_{state}'] = memory_update_in
            successful_states.append(state)
        else:
            print(f"   └─ ❌ Failed: فشل في الثقة.")

    accumulative_conf += best_confidence
    print(f"\n🎯 Best State: {best_state} | Accuracy: {best_confidence*100:.1f}% | Avg: {(accumulative_conf/max(1, step_idx))*100:.1f}%")
    print(f"📊 Predicted angles: {np.round(best_predicted_angles, 2)}")
    
    if successful_states:
        np.savez_compressed(weights_file, **updated_weights)
        print(f"💾 Saved weights for: {successful_states}")

    return True, memory_state, best_predicted_angles, accumulative_conf