import numpy as np
import sys
import os

# إضافة المجلد الحالي لمسار البحث لضمان عمل الاستيرادات
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train.online_learning import process_online_sample

def main():
    print("="*60)
    print("🌡️ Sensation Interface: Thermal Input Mode 🌡️")
    print("="*60)

    memory_state = {'last_delta_in': 0, 'last_delta_out': 0}
    accumulative_conf = 0.0
    step = 1

    while True:
        print("\n" + "-"*60)
        user_input = input("Enter 12 temperature readings (space separated) \n> ")
        
        if user_input.lower() == 'exit': break
            
        try:
            values = [float(x) for x in user_input.split()]
            if len(values) != 12:
                print(f"⚠️ Error: You entered {len(values)} values, need 12.")
                continue
            
            # 1. تسوية الداتا (Min-Max Normalization)
            raw_row = np.array(values)
            min_v, max_v = np.min(raw_row), np.max(raw_row)
            norm_row = (raw_row - min_v) / (max_v - min_v + 1e-8)
            
            # 2. إنشاء نافذة زمنية (200 عينة لـ 12 قناة)
            analog_input = np.tile(norm_row, (200, 1))
            analog_input += np.random.normal(0, 0.01, analog_input.shape)
            
            # 3. المعالجة في الشبكة العصبية
            success, memory_state, stim, accumulative_conf = process_online_sample(
                None, analog_input, memory_state, step, accumulative_conf
            )

            if success and stim is not None:
                v, c, f = stim
                print(f"\n⚡ ELECTROTACTILE COMMAND:")
                print(f"   ├─ Voltage:   {v:.2f} V")
                print(f"   ├─ Current:   {c:.2f} mA")
                print(f"   └─ Frequency: {f:.2f} Hz")

            step += 1
            
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()