import numpy as np
import sys
import os
import serial 
import time

# 1. ربط المسار بملف الأونلاين ليرنينج
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train.online_learning import process_online_sample

def main():
    try:
        # timeout=None عشان يستنى الإشارة براحته وميستعجلش
        ser = serial.Serial('COM2', 9600, timeout=None)
        time.sleep(2) 
        print("connected to serial port COM2 at 9600 baud.")
    except Exception as e:
        print(f"serial error {e}")
        return

    memory_state = {}
    accumulative_conf = 0.0
    step = 1

    while True:
        try:
            # --- الخطوة 1: استقبال إشارة واحدة فقط ---
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            
            values = [float(x) for x in line.split()]
            
            if len(values) == 12:
                print("\n" + "="*60)
                print(f"reciving signal ({step}): {values}")
                
                # --- الخطوة 2: تجهيز الإشارة (تكرارها لـ 200 لعمل نافذة للشبكة) ---
                raw_row = np.array(values)
                min_v, max_v = np.min(raw_row), np.max(raw_row)
                norm_row = (raw_row - min_v) / (max_v - min_v + 1e-8)
                
                analog_input = np.tile(norm_row, (200, 1))
                analog_input += np.random.normal(0, 0.01, analog_input.shape)
                print("signal prepared for processing.")
                
                # --- الخطوة 3: إرسالها لملف online_learning (اختبار الثقة وتعديل الأوزان) ---
                print("operations in progress...")
                success, memory_state, stim, accumulative_conf = process_online_sample(
                    None, analog_input, memory_state, step, accumulative_conf
                )

                # --- الخطوة 4: طباعة الناتج النهائي ---
                if success and stim is not None:
                    v, c, f = stim
                    print(f" the result-> V: {v:.1f}V | I: {c:.1f}mA | F: {f:.1f}Hz")
                else:
                    print("signal is weak or no spikes generated, skipping output.")
                
                print("⏳ waiting for signal.")
                step += 1
                
        except Exception as e:
            print(f"error {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()