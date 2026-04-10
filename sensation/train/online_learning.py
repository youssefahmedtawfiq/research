import sys
import os
from brian2 import *
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../")) 
if project_root not in sys.path:
    sys.path.append(project_root)

signal_proc_dir = os.path.join(project_root, "signalRead_Procescing")
if signal_proc_dir not in sys.path:
    sys.path.append(signal_proc_dir)

import inout as inout
from model.network import create_network
from signalRead_Procescing.encoding import rate_encoding  

def process_and_encode_window(analog_window):
    segment = analog_window.reshape(1, 200, 12)
    segment_01 = np.clip(segment, 0, 1)
    rate_spikes = rate_encoding(segment_01, time_steps=20, random_seed=42)
    time_idx, channel_idx = np.where(rate_spikes[0] > 0)
    
    indices = channel_idx 
    times = time_idx * 10.0 
    return indices, times

def decode_spikes_to_stimulus(spike_counts):
    # 🛑 وسعنا الميزان هنا للنيورون الواحد 🛑
    # بدل ما كان بيوصل للماكس بعد 30 أو 150 نبضة، خليناه 300 نبضة
    # ده هيخلي الفولت والتيار يتدرجوا بنعومة مع قوة الإشارة
    raw = np.clip(spike_counts / 300.0, 0, 1)
    v = raw[0] * 50.0        
    c = raw[1] * 10.0        
    f = 50 + (raw[2] * 150)  
    return [v, c, f]

def process_online_sample(net_objects, analog_thermal_input, memory_state, step_idx, accumulative_conf):
    alpha_memory, learning_rate = 0.6, 0.05
    
    indices, times = process_and_encode_window(analog_thermal_input)
    if len(indices) == 0: 
        return False, memory_state, None, accumulative_conf
        
    weights_file = os.path.join(project_root, "processed_data/feedback_network_weights.npz")
    
    try:
        updated_weights = dict(np.load(weights_file))
        weights_found = True
        print(" old weights loaded successfully.")
    except:
        updated_weights, weights_found = {}, False
        print("  No saved weights found. Starting with random initialization.")

    best_raw_score = -1.0
    best_state = 1
    best_stim = None
    best_conf_display = 0.0
    successful_states = []
    
    print("5 states processing (Step: {})".format(step_idx))
    
    for state in range(1, 6): 
        start_scope() 
        inout.NEURON_STATE = state
        seed(42 + state * 10) 
        
        inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h = create_network()
        out_mon = SpikeMonitor(out_g)
        temp_net = Network(inp_g, hid_g, out_g, S_i_h, S_h_o, S_h_h, out_mon)
        
        if weights_found:
            w_in = updated_weights.get(f'w_in_{state}')
            w_out = updated_weights.get(f'w_out_{state}')
            
            if w_in is not None and len(w_in) == len(S_i_h.w):
                S_i_h.w = w_in
            if w_out is not None and len(w_out) == len(S_h_o.w):
                S_h_o.w = w_out
            
        inp_g.set_spikes(indices, times * ms)
        temp_net.run(200*ms)
        
        counts = np.array(out_mon.count)
        stim = decode_spikes_to_stimulus(counts)
        
        total_spikes = np.sum(counts)
        
        # 🛑 وسعنا ميزان الثقة الإجمالية للشبكة 🛑
        # خلينا المقام 1000، يعني الشبكة لازم تجيب 1000 نبضة عشان توصل لـ 100%
        # ده هيخليك تشوف نسب زي 40% و 65% و 80% وتقدر تقارن بينهم صح
        raw_score = total_spikes / 1000.0 
        conf_display = min(raw_score, 1.0) if total_spikes > 10 else 0.0
            
        status = "accepted" if conf_display >= 0.10 else "❌ مرفوضة"
        print(f"      ├─ State {state}: confidence percent= {conf_display*100:.1f}% ({status}) (Spikes: {total_spikes})")
        
        if raw_score > best_raw_score:
            best_raw_score = raw_score
            best_state = state
            best_stim = stim
            best_conf_display = conf_display

        if conf_display >= 0.10:
            updated_weights[f'w_in_{state}'] = np.array(S_i_h.w)
            updated_weights[f'w_out_{state}'] = np.array(S_h_o.w)
            successful_states.append(state)

    if successful_states:
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        np.savez_compressed(weights_file, **updated_weights)
        print(f"  the best state is {best_state} (highest confidence {best_conf_display*100:.1f}%)weights updated for states: {successful_states}")
    else:
        print(" no states passed the confidence threshold. No weights updated.")

    return True, memory_state, best_stim, accumulative_conf + best_conf_display