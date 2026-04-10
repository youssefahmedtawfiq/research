import numpy as np
from train.online_learning import process_online_sample
from motor_drive import JointServo # استدعاء المحرك الجديد
import serial
# حط السطر ده قبل الـ loop بتاعتك
ser = serial.Serial('COM6', 9600, timeout=None)
def main():
    
    print("="*60)
    print("🟢 وضع الإدخال اليدوي (Raw Dataset Mode) 🟢")
    print("="*60)

    memory_state = {'last_delta_in': 0, 'last_delta_out': 0}
    accumulative_conf = 0.0
    step = 1
    my_joint = JointServo()
    while True:
        print("\n" + "-"*60)
       # السطر الجديد:
        user_input = ser.readline().decode('utf-8').strip()
        if not user_input: continue # عشان لو السطر فاضي ميعملش إيرور
        
        if user_input.lower() == 'exit':
            break
            
        try:
            values = [float(x) for x in user_input.split()]
            if len(values) != 12:
                print(f"⚠️ خطأ: أدخلت {len(values)} أرقام، المطلوب 12.")
                continue
            
            # 🌟 السحر هنا: تسوية الداتا لمنع العمى والتشبع 🌟
            raw_row = np.array(values)
            min_val = np.min(raw_row)
            max_val = np.max(raw_row)
            
            if max_val > min_val:
                normalized_row = (raw_row - min_val) / (max_val - min_val)
            else:
                normalized_row = np.zeros_like(raw_row)
            
            # تكرار النمط 200 مرة للشبكة
            base_row = normalized_row.reshape(1, -1)
            analog_input = np.repeat(base_row, 200, axis=0)
            noise = np.random.normal(0, 1e-4, analog_input.shape)
            analog_input = analog_input + noise
            
            success, memory_state, output_angles, accumulative_conf = process_online_sample(
                net_objects=None, analog_emg_input=analog_input,
                memory_state=memory_state, step_idx=step, accumulative_conf=accumulative_conf
            )
            if success:
            # 🌟 الربط مع معادلة المفصل:
            # نأخذ القيمة المتوسطة للزوايا المتوقعة كعزم دوران (Torque)
                torque_signal = np.mean(output_angles) 
            
            # تحديث حركة المفصل الفيزيائية
                actual_angle = my_joint.update(torque_signal)
            
                print(f"\n⚙️ Servo Driver Update:")
                print(f"   ├─ Control Signal (Torque): {torque_signal:.4f}")
                print(f"   └─ Actual Physical Angle: {actual_angle:.2f}°")
            step += 1
            
        except ValueError:
            print("⚠️ خطأ في الأرقام.")

if __name__ == "__main__":
    main()