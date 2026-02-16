from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import time
import requests
import os

app = Flask(__name__)

# ===== CONFIG =====
BOT_TOKEN = "8217700733:AAGgdc8yEXlaKKt6CtfY4RO-yjSyAUJFF2g"
CHAT_ID = "8417938771"

# โหลด Model (ใส่ compile=False เพื่อเลี่ยง Warning ที่คุณเจอตอนแรก)
MODEL = tf.keras.models.load_model("keras_model.h5", compile=False)
LABELS = open("labels.txt").read().splitlines()

# ตัวแปรสถานะ
class2_start = None
last_seen = None
telegram_sent = False

# ===== หน้าแรก (ป้องกัน 404) =====
@app.route('/')
def home():
    return "<h1>AI Server is Live!</h1><p>Send images to /upload to process.</p>"

# ===== TELEGRAM FUNCTION =====
def send_telegram(prob):
    msg = f"⚠️ ALERT\nEyes closing detected!\nConfidence: {prob*100:.1f}%\nDuration: >10s"
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ===== RECEIVE IMAGE & PROCESS =====
@app.route("/upload", methods=["POST"])
def upload():
    global class2_start, last_seen, telegram_sent

    if 'image' not in request.files:
        return "No image file", 400

    try:
        # อ่านไฟล์ภาพ
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Invalid image", 400

        # เตรียมภาพสำหรับ Model
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1 # ปรับให้ตรงกับ Teachable Machine
        img_final = np.expand_dims(img_normalized, axis=0)

        # ทำนายผล
        pred = MODEL.predict(img_final)[0]
        now = time.time()

        detected = False
        prob = 0

        # ตรวจสอบ Label (อิงตามชื่อใน labels.txt)
        for i, p in enumerate(pred):
            label_name = LABELS[i].strip().split(' ', 1)[-1] # ตัดตัวเลขข้างหน้าออกถ้ามี
            if label_name == "eyes_close" and p >= 0.7: # ปรับความแม่นยำเป็น 70%
                detected = True
                prob = p

        # ตรรกะการจับเวลา (Drowsiness Logic)
        if detected:
            if class2_start is None:
                class2_start = now
            last_seen = now
            
            # ถ้าหลับตาค้างเกิน 10 วินาที และยังไม่ได้ส่งแจ้งเตือน
            duration = now - class2_start
            if duration >= 10 and not telegram_sent:
                send_telegram(prob)
                telegram_sent = True
                print("!!! ALERT SENT TO TELEGRAM !!!")
        else:
            # ถ้าลืมตาแล้ว ให้รีเซ็ตสถานะ
            class2_start = None
            last_seen = None
            telegram_sent = False

        return jsonify({
            "status": "ok", 
            "detected": detected, 
            "confidence": float(prob),
            "duration": float(now - class2_start) if class2_start else 0
        })

    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
