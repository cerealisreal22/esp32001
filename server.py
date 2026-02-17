from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import cv2
import time
import requests
import os
import base64

app = Flask(__name__)

# ===== CONFIG =====
BOT_TOKEN = "8217700733:AAGgdc8yEXlaKKt6CtfY4RO-yjSyAUJFF2g"
CHAT_ID = "8417938771"

# โหลด Model
MODEL = tf.keras.models.load_model("keras_model.h5", compile=False)
LABELS = open("labels.txt").read().splitlines()

# ตัวแปรระบบ
class2_start = None
telegram_sent = False
last_data = {
    "image_base64": "",
    "confidence": 0,
    "detected": False,
    "duration": 0,
    "last_update": "Waiting..."
}

# ===== HTML DASHBOARD (หน้าแรก) =====
@app.route('/')
def home():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Drowsiness Monitor</title>
        <meta http-equiv="refresh" content="2"> <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; background: #eceff1; color: #37474f; }
            .card { background: white; padding: 20px; border-radius: 15px; display: inline-block; margin-top: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-top: 5px solid #2196f3; }
            img { width: 320px; height: 240px; border-radius: 10px; border: 3px solid #cfd8dc; background: #000; }
            .data { margin-top: 15px; font-size: 1.1em; text-align: left; }
            .status-box { padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 10px; }
            .normal { background: #c8e6c9; color: #2e7d32; }
            .alert { background: #ffcdd2; color: #c62828; animation: blinker 1s linear infinite; }
            @keyframes blinker { 50% { opacity: 0.5; } }
        </style>
    </head>
    <body>
        <h2>👁️ AI Drowsiness Dashboard</h2>
        <div class="card">
            <div class="status-box {{ 'alert' if detected else 'normal' }}">
                {{ '⚠️ SLEEP DETECTED' if detected else '✅ NORMAL' }}
            </div>
            {% if image_base64 %}
                <img src="data:image/jpeg;base64,{{ image_base64 }}">
            {% else %}
                <div style="width:320px; height:240px; background:#ccc; line-height:240px;">No Image</div>
            {% endif %}
            <div class="data">
                <p><b>Confidence:</b> {{ (confidence * 100)|round(2) }}%</p>
                <p><b>Duration:</b> {{ duration|round(2) }} sec</p>
                <p><b>Last Update:</b> {{ last_update }}</p>
            </div>
        </div>
        <p><small>Refreshing every 2 seconds</small></p>
    </body>
    </html>
    """
    return render_template_string(html_template, **last_data)

# ===== TELEGRAM FUNCTION =====
def send_telegram(prob):
    msg = f"⚠️ ALERT: Drowsiness Detected!\nConfidence: {prob*100:.1f}%\nStatus: Eyes closed > 10s"
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ===== RECEIVE & PROCESS IMAGE =====
@app.route("/upload", methods=["POST"])
def upload():
    global class2_start, telegram_sent, last_data

    if 'image' not in request.files:
        return "Missing image", 400

    try:
        file = request.files["image"]
        img_raw = file.read()
        
        # เก็บภาพลง Dashboard (Base64)
        last_data["image_base64"] = base64.b64encode(img_raw).decode('utf-8')

        # ประมวลผลภาพ
        img_bytes = np.frombuffer(img_raw, np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None: return "Invalid Image", 400

        img_resized = cv2.resize(img, (224, 224))
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
        img_final = np.expand_dims(img_normalized, axis=0)

        # AI Predict
        pred = MODEL.predict(img_final)[0]
        now = time.time()
        
        detected = False
        prob = 0
        for i, p in enumerate(pred):
            label = LABELS[i].strip().split(' ', 1)[-1]
            if label == "eyes_close" and p >= 0.7:
                detected = True
                prob = p

        # Logic การจับเวลา
        if detected:
            if class2_start is None: class2_start = now
            duration = now - class2_start
            if duration >= 10 and not telegram_sent:
                send_telegram(prob)
                telegram_sent = True
        else:
            class2_start = None
            telegram_sent = False
            duration = 0

        # อัปเดตข้อมูลหน้าเว็บ
        last_data.update({
            "confidence": float(prob) if detected else float(max(pred)),
            "detected": detected,
            "duration": float(duration),
            "last_update": time.strftime("%H:%M:%S")
        })

        return "OK"

    except Exception as e:
        print(f"Server Error: {e}")
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
