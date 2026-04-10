// متغيرات للتحكم في وقت الإرسال
unsigned long lastSendTime = 0;
const int sendInterval = 500; // 🛑 التعديل هنا: إرسال قراءة كل 500 ملي ثانية (نص ثانية) 🛑

// متغيرات لمحاكاة حالة العضلة
unsigned long lastStateChange = 0;
bool isMuscleActive = false; // هل العضلة منقبضة أم في حالة راحة؟

void setup() {
  // نفس الـ Baud Rate بتاع البايثون والبروتس
  Serial.begin(9600);
  
  // تغذية مولد الأرقام العشوائية من طرف تناظري غير متصل لزيادة العشوائية
  randomSeed(analogRead(0));
}

void loop() {
  // ==========================================
  // 1. تغيير حالة العضلة كل فترة (محاكاة حركة اليد)
  // ==========================================
  // تغيير الحالة كل 2 إلى 5 ثواني
  if (millis() - lastStateChange > random(2000, 5000)) {
    isMuscleActive = !isMuscleActive;
    lastStateChange = millis();
  }

  // ==========================================
  // 2. إرسال 12 قناة (Channels) عبر السيريال
  // ==========================================
  if (millis() - lastSendTime >= sendInterval) {
    for (int i = 0; i < 12; i++) {
      float emgSignal = 0.0;

      if (isMuscleActive) {
        // حالة الانقباض: أرقام عالية وتذبذب عالي (تحاكي الـ NINAPRO Spikes)
        if (i % 3 == 0) {
          emgSignal = random(400, 900) / 10.0; // قنوات متأثرة جداً (40.0 إلى 90.0)
        } else {
          emgSignal = random(200, 600) / 10.0; // قنوات متأثرة بشكل متوسط
        }
      } else {
        // حالة الراحة: شوشرة خفيفة (Baseline Noise)
        emgSignal = random(50, 150) / 10.0; // أرقام هادية (5.0 إلى 15.0)
      }

      // طباعة الرقم مع مسافة
      Serial.print(emgSignal, 1); 
      if (i < 11) {
        Serial.print(" ");
      }
    }
    
    // سطر جديد بعد الـ 12 رقم
    Serial.println();
    
    lastSendTime = millis();
  }
}