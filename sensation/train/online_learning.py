import sys
import os
from brian2 import *
import numpy as np

# حل مشكلة المسارات: جعل مجلد sensation هو المجلد الرئيسي للبحث
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../")) # المجلد الرئيسي sensation
if project_root not in sys.path:
    sys.path.append(project_root)

# استيراد من المجلدات المحلية مباشرة
import inout as inout
from signalRead_Procescing.emg_ninapro_db2_subject1 import process_and_encode_window
from model.network import create_network

def decode_spikes_to_stimulus(spike_counts):
    """تحويل النبضات لثلاث قيم فيزيائية للجهاز"""
    raw = np.clip(spike_counts / 30.0, 0, 1)
    v = raw[0] * 50.0        # Volt: 0-50V
    c = raw[1] * 10.0        # Current: 0-10mA
    f = 50 + (raw[2] * 150)  # Frequency: 50-200Hz
    return [v, c, f]

def process_online_sample(net_objects, analog_thermal_input, memory_state, step_idx, accumulative_conf):
    alpha_memory, learning_rate = 0.6, 0.05
    
    # 1. التشفير
    indices, times = process_and_encode_window(analog_thermal_input)
    if len(indices) == 0: return False, memory_state, None, accumulative_conf
        
    weights_file = os.path.join(project_root, "processed_data/feedback_network_weights.npz")
    try:
        updated_weights = dict(np.load(weights_file))
        weights_found = True
    except:
        updated_weights, weights_found = {}, False

    best_state, best_conf, best_stim = 1, -1.0, None
    successful_states = []
    
    print(f"\n🌡️ Processing Feedback Loop (Step: {step_idx})")
    
    for state in range(1, 4): # تجربة 3 مستويات للحرارة
        inout.NEURON_STATE = state
        seed(42)
        
        # إنشاء الشبكة (تأكد أن المخرج 3 نيورونات في network.py)
        inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h = create_network()
        out_mon = SpikeMonitor(out_g)
        temp_net = Network(inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h, out_mon)
        
        if weights_found:
            S_i_h.w = updated_weights.get(f'w_in_{state}', S_i_h.w)
            S_h_o.w = updated_weights.get(f'w_out_{state}', S_h_o.w)
            
        inp_g.set_spikes(indices, times * ms)
        temp_net.run(200*ms)
        
        counts = np.array(out_mon.count)
        stim = decode_spikes_to_stimulus(counts)
        conf = min((np.sum(counts) / 90.0), 1.0) if np.sum(counts) > 2 else 0.0
            
        if conf > best_conf:
            best_conf, best_state, best_stim = conf, state, stim

        if conf >= 0.10:
            updated_weights[f'w_in_{state}'] = np.array(S_i_h.w)
            updated_weights[f'w_out_{state}'] = np.array(S_h_o.w)
            successful_states.append(state)

    if successful_states:
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        np.savez_compressed(weights_file, **updated_weights)

    return True, memory_state, best_stim, accumulative_conf + best_conf