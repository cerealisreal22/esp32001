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

MODEL = tf.keras.models.load_model("keras_model.h5", compile=False)
LABELS = open("labels.txt").read().splitlines()

class2_start = None
telegram_sent = False
last_data = {
    "image_base64": "",
    "closed_prob": 0,
    "open_prob": 0,
    "detected": False,
    "duration": 0,
    "last_update": "Waiting..."
}

@app.route('/')
def home():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Drowsiness Monitor</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body { font-family: sans-serif; text-align: center; background: #eceff1; padding: 20px; }
            .card { background: white; padding: 20px; border-radius: 15px; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            img { width: 320px; border-radius: 10px; border: 2px solid #ccc; }
            .prob-bar { margin: 10px 0; text-align: left; background: #eee; border-radius: 5px; overflow: hidden; }
            .fill { height: 20px; line-height: 20px; color: white; padding-left: 10px; font-size: 0.8em; }
            .closed { background: #f44336; }
            .open { background: #4caf50; }
            .status { font-weight: bold; margin-bottom: 10px; padding: 10px; border-radius: 5px; }
            .alert { background: #ffcdd2; color: #b71c1c; }
            .normal { background: #c8e6c9; color: #1b5e20; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="status {{ 'alert' if detected else 'normal' }}">
                {{ '⚠️ SLEEPING' if detected else '✅ AWAKE' }}
            </div>
            <img src="data:image/jpeg;base64,{{ image_base64 }}">
            
            <div style="margin-top:15px;">
                <div class="prob-bar">
                    <div class="fill closed" style="width: {{ (closed_prob * 100)|round }}%">Eyes Closed: {{ (closed_prob * 100)|round(1) }}%</div>
                </div>
                <div class="prob-bar">
                    <div class="fill open" style="width: {{ (open_prob * 100)|round }}%">Eyes Open: {{ (open_prob * 100)|round(1) }}%</div>
                </div>
            </div>
            
            <p>Duration: {{ duration|round(1) }} sec | Update: {{ last_update }}</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, **last_data)

@app.route("/upload", methods=["POST"])
def upload():
    global class2_start, telegram_sent, last_data
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
    
    # ดึงค่าตาม Label (สมมติ 0: open, 1: close หรือตามลำดับใน labels.txt)
    # เราจะหาค่าจากชื่อ label ตรงๆ เพื่อความชัวร์
    probs = {}
    for i, p in enumerate(pred):
        label = LABELS[i].strip().split(' ', 1)[-1]
        probs[label] = float(p)

    c_prob = probs.get("eyes_close", 0)
    o_prob = probs.get("eyes_open", 0)

    detected = c_prob >= 0.7
    duration = 0

    if detected:
        if class2_start is None: class2_start = now
        duration = now - class2_start
        if duration >= 10 and not telegram_sent:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                          json={"chat_id": CHAT_ID, "text": f"⚠️ ALERT!\nEyes Closed: {c_prob*100:.1f}%"}, timeout=5)
            telegram_sent = True
    else:
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
