# inout.py

# إعدادات الشبكة
INPUT_NEURONS = 4
OUTPUT_NEURONS = 2

# اختيار حالة Izhikevich من 1 إلى 5
# 1: RS, 2: IB, 3: CH, 4: FS, 5: LTS
NEURON_STATE = 1  

# بيانات وهمية للتدريب (Input Rates بالـ Hz و Target Neuron Index)
train_data = [
    {'input': [200, 0, 200, 0], 'target': 0},
    {'input': [0, 200, 0, 200], 'target': 1},
    {'input': [150, 50, 150, 0], 'target': 0},
    {'input': [50, 150, 0, 150], 'target': 1}
]