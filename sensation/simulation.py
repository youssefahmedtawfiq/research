# simulation.py
import time
import numpy as np
import os
from train.online_learning import process_online_sample

def hardware_profiler(cpu_time_sec):
    """
    دالة لحساب ومقارنة استهلاك الطاقة والزمن بين الـ CPU والـ Neuromorphic Chip
    """
    cpu_power_watts = 25.0 
    cpu_energy_joules = cpu_power_watts * cpu_time_sec
    cpu_energy_mj = cpu_energy_joules * 1000 

    neuro_speedup = 250 
    neuro_time_sec = cpu_time_sec / neuro_speedup

    neuro_static_power_watts = 0.1 
    neuro_dynamic_energy_mj = 1.5 
    neuro_energy_mj = (neuro_static_power_watts * neuro_time_sec * 1000) + neuro_dynamic_energy_mj

    return {
        'cpu_time_ms': cpu_time_sec * 1000,
        'cpu_energy_mj': cpu_energy_mj,
        'neuro_time_ms': neuro_time_sec * 1000,
        'neuro_energy_mj': neuro_energy_mj,
        'speedup': neuro_speedup,
        'energy_saving': cpu_energy_mj / neuro_energy_mj if neuro_energy_mj > 0 else 0
    }

def run_simulation(user_inputs_list):

    print("(Automated Simulation Mode) ")
   

    memory_state = {'last_delta_in': 0, 'last_delta_out': 0}
    accumulative_conf = 0.0
    
    # 🌟 1. تجهيز قاموس (Dictionary) لتجميع بيانات الطاقة والسرعة أثناء المحاكاة 🌟
    metrics_history = {
        'cpu_time_ms': [],
        'cpu_energy_mj': [],
        'neuro_time_ms': [],
        'neuro_energy_mj': [],
        'speedup': [],
        'energy_saving': []
    }
    
    for step, values in enumerate(user_inputs_list, start=1):
       
        print(f"جاري معالجة الإدخال رقم {step}...")
        
        # تجهيز الداتا
        raw_row = np.array(values)
        min_val = np.min(raw_row)
        max_val = np.max(raw_row)
        
        if max_val > min_val:
            normalized_row = (raw_row - min_val) / (max_val - min_val)
        else:
            normalized_row = np.zeros_like(raw_row)
            
        print("data normalized")
        
        base_row = normalized_row.reshape(1, -1)
        analog_input = np.repeat(base_row, 200, axis=0)
        noise = np.random.normal(0, np.abs(base_row) * 0.001 + 1e-9, analog_input.shape)
        analog_input = analog_input + noise
        
        # التوقيت والمعالجة
        start_time = time.perf_counter()
        
        success, memory_state, output_angles, accumulative_conf = process_online_sample(
            net_objects=None,
            analog_thermal_input=analog_input,
            memory_state=memory_state,
            step_idx=step,
            accumulative_conf=accumulative_conf
        )
        
        cpu_time_sec = time.perf_counter() - start_time
        hw_profile = hardware_profiler(cpu_time_sec)
        
        # 🌟 2. تسجيل البيانات الحالية في القاموس 🌟
        metrics_history['cpu_time_ms'].append(hw_profile['cpu_time_ms'])
        metrics_history['cpu_energy_mj'].append(hw_profile['cpu_energy_mj'])
        metrics_history['neuro_time_ms'].append(hw_profile['neuro_time_ms'])
        metrics_history['neuro_energy_mj'].append(hw_profile['neuro_energy_mj'])
        metrics_history['speedup'].append(hw_profile['speedup'])
        metrics_history['energy_saving'].append(hw_profile['energy_saving'])
        
        # الطباعة
       
        print("   Hardware Implementation Profile (CPU vs Neuromorphic)   ")
        
        print(f" CPU Processor:")
        print(f"   ├─ Latency (Speed): {hw_profile['cpu_time_ms']:.2f} ms")
        print(f"   └─ Energy Consumed: {hw_profile['cpu_energy_mj']:.2f} mJ")

        print(f"\n Neuromorphic Chip (Estimated Loihi):")
        print(f"   ├─ Latency (Speed): {hw_profile['neuro_time_ms']:.2f} ms")
        print(f"   └─ Energy Consumed: {hw_profile['neuro_energy_mj']:.2f} mJ")

        print(f"\n Results: Neuromorphic is {hw_profile['speedup']:.0f}x Faster and {hw_profile['energy_saving']:.0f}x more Energy-Efficient!")
        

    # 🌟 3. حفظ مصفوفات الطاقة في ملف npz. بشكل تراكمي 🌟
    metrics_file = "processed_data/hardware_metrics.npz"
    os.makedirs("processed_data", exist_ok=True) # التأكد إن الفولدر موجود

    try:
        # لو الملف موجود من تجارب قبل كده، افتحه
        existing_metrics = dict(np.load(metrics_file, allow_pickle=True))
        # ضيف الداتا الجديدة على الداتا القديمة
        for key in metrics_history:
            existing_metrics[key] = np.concatenate((existing_metrics.get(key, []), metrics_history[key]))
    except:
        # لو الملف مش موجود، اعمل واحد جديد
        existing_metrics = {key: np.array(val) for key, val in metrics_history.items()}

    np.savez_compressed(metrics_file, **existing_metrics)
    print(f"\n🔋 تم حفظ سجلات الطاقة والزمن بنجاح في: {metrics_file}")

if __name__ == "__main__":
    test_data = [
       [0.0000017214358, 0.0000014072614, 0.00000053924919, -0.0000029182454, -0.00000031468321, 0.0000060771486, 0.000010270136, 0.0000067508231, 0.0000034963398, 0.000014480399, -0.0000020452155, 0.0000094638763]
       ]
    
    run_simulation(test_data)