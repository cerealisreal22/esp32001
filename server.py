from flask import Flask, request, jsonify, render_template_string, redirect, url_for
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

MODEL = tf.keras.models.load_model("keras_model.h5", compile=False)
LABELS = open("labels.txt").read().splitlines()

# ตัวแปรระบบ
system_enabled = False  
class2_start = None
telegram_sent = False
last_data = {
    "image_base64": "",
    "closed_prob": 0,
    "open_prob": 0,
    "detected": False,
    "duration": 0,
    "last_update": "Waiting...",
    "system_enabled": False
}

# --- ฟังก์ชันดึงพิกัดจาก IP ---
def get_location_link(ip):
    try:
        # ใช้ ip-api.com (ฟรี ไม่ต้องใช้คีย์)
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
        if response.get("status") == "success":
            lat = response.get("lat")
            lon = response.get("lon")
            city = response.get("city")
            return f"\n📍 Location: {city}\n🔗 View on Maps: https://www.google.com/maps?q={lat},{lon}"
    except:
        pass
    return "\n📍 Location: Unknown (GPS module missing)"

@app.route('/')
def home():
    last_data["system_enabled"] = system_enabled
    # (HTML คงเดิมจากที่คุณมี)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Monitor (With Switch)</title>
        <meta http-equiv="refresh" content="3">
        <style>
            body { font-family: sans-serif; text-align: center; background: #eceff1; padding: 20px; }
            .card { background: white; padding: 20px; border-radius: 15px; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            img { width: 320px; border-radius: 10px; border: 2px solid #ccc; }
            .btn { padding: 10px 30px; font-size: 1.2em; cursor: pointer; border-radius: 50px; border: none; color: white; transition: 0.3s; margin-bottom: 20px; }
            .btn-on { background: #4caf50; box-shadow: 0 4px #2e7d32; }
            .btn-off { background: #f44336; box-shadow: 0 4px #b71c1c; }
            .prob-bar { margin: 10px 0; text-align: left; background: #eee; border-radius: 5px; overflow: hidden; }
            .fill { height: 20px; line-height: 20px; color: white; padding-left: 10px; font-size: 0.8em; transition: 0.5s; }
            .closed { background: #f44336; }
            .open { background: #4caf50; }
            .status { font-weight: bold; margin-bottom: 15px; padding: 10px; border-radius: 5px; }
            .alert { background: #ffcdd2; color: #b71c1c; animation: blink 1s infinite; }
            .normal { background: #c8e6c9; color: #1b5e20; }
            .disabled { background: #e0e0e0; color: #757575; }
            @keyframes blink { 50% { opacity: 0.6; } }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>System Control</h2>
            <form action="/toggle" method="POST">
                {% if system_enabled %}
                    <button type="submit" class="btn btn-off">STOP MONITORING (ON)</button>
                {% else %}
                    <button type="submit" class="btn btn-on">START MONITORING (OFF)</button>
                {% endif %}
            </form>
            <div class="status {{ 'alert' if detected and system_enabled else ('normal' if system_enabled else 'disabled') }}">
                {% if not system_enabled %} SYSTEM PAUSED {% elif detected %} ⚠️ SLEEPING DETECTED {% else %} ✅ MONITORING: AWAKE {% endif %}
            </div>
            <img src="data:image/jpeg;base64,{{ image_base64 }}">
            <div style="margin-top:15px;">
                <div class="prob-bar"><div class="fill closed" style="width: {{ (closed_prob * 100)|round }}%">Closed: {{ (closed_prob * 100)|round(1) }}%</div></div>
                <div class="prob-bar"><div class="fill open" style="width: {{ (open_prob * 100)|round }}%">Open: {{ (open_prob * 100)|round(1) }}%</div></div>
            </div>
            <p>Duration: {{ duration|round(1) }} sec | Update: {{ last_update }}</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, **last_data)

@app.route('/toggle', methods=['POST'])
def toggle():
    global system_enabled, class2_start, telegram_sent
    system_enabled = not system_enabled
    class2_start = None
    telegram_sent = False
    return redirect(url_for('home'))

@app.route("/upload", methods=["POST"])
def upload():
    global class2_start, telegram_sent, last_data, system_enabled
    
    file = request.files["image"]
    img_raw = file.read()
    last_data["image_base64"] = base64.b64encode(img_raw).decode('utf-8')

    img_bytes = np.frombuffer(img_raw, np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
    img_final = np.expand_dims(img_normalized, axis=0)

    pred = MODEL.predict(img_final)[0]
    now = time.time()
    
    probs = {}
    for i, p in enumerate(pred):
        label = LABELS[i].strip().split(' ', 1)[-1]
        probs[label] = float(p)

    c_prob = probs.get("eyes_close", 0)
    o_prob = probs.get("eyes_open", 0)
    detected = c_prob >= 0.7
    duration = 0

    if system_enabled and detected:
        if class2_start is None: class2_start = now
        duration = now - class2_start
        if duration >= 10 and not telegram_sent:
            try:
                # --- ส่วนดึง IP และ พิกัด ---
                client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                location_msg = get_location_link(client_ip)
                
                alert_text = f"⚠️ ALERT!\nEyes Closed: {c_prob*100:.1f}%\nStatus: Sleeping Detected!{location_msg}"
                
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                              json={"chat_id": CHAT_ID, "text": alert_text}, timeout=5)
                telegram_sent = True
            except Exception as e:
                print(f"Telegram Error: {e}")
    elif not detected or not system_enabled:
        class2_start = None
        telegram_sent = False

    last_data.update({
        "closed_prob": c_prob,
        "open_prob": o_prob,
        "detected": detected,
        "duration": duration,
        "last_update": time.strftime("%H:%M:%S")
    })
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
