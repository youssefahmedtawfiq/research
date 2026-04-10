# joint_driver.py
import numpy as np

class JointServo:
    def __init__(self):
        # المعاملات الفيزيائية من معادلة (28) في البحث
        self.I = 0.05      # القصور الذاتي (Inertia)
        self.b = 0.1       # معامل الاحتكاك (Damping)
        self.mgL = 0.5     # عزم الجاذبية (Gravity)
        
        # الحالة الحالية للمفصل
        self.theta = 0.0      # الزاوية الحالية (راديان)
        self.theta_dot = 0.0  # السرعة الزاوية الحالية
        self.dt = 0.05        # الزمن المستغرق (50ms - Window duration)

    def update(self, snn_output_torque):
        """
        تطبيق معادلة الديناميكا لتحديث الزاوية
        """
        # حساب التسارع الزاوي (theta double dot) بناءً على المنهجية
        # Torque_SNN = snn_output_torque
        theta_double_dot = (snn_output_torque - self.b * self.theta_dot - self.mgL * np.sin(self.theta)) / self.I
        
        # التكامل العددي (Euler Integration) للحصول على السرعة والزاوية الجديدة
        self.theta_dot += theta_double_dot * self.dt
        self.theta += self.theta_dot * self.dt
        
        # تحويل النتيجة لدرجات لسهولة القراءة
        return np.degrees(self.theta)