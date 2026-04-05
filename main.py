# main.py
from brian2 import prefs
from train.train import train_network

# إعداد لعدم استخدام مترجم C++ لضمان تشغيل الكود بسلاسة على أي جهاز بايثون
prefs.codegen.target = 'numpy'

if __name__ == '__main__':
    print("========================================")
    print(" Spiking Neural Network (Izhikevich)    ")
    print("========================================")
    train_network()