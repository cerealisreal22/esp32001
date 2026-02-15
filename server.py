from flask import Flask, request
import tensorflow as tf
import numpy as np
import cv2
import time
import requests

app = Flask(__name__)

# ===== CONFIG =====
BOT_TOKEN = "8217700733:AAGgdc8yEXlaKKt6CtfY4RO-yjSyAUJFF2g"
CHAT_ID = "8417938771"

MODEL = tf.keras.models.load_model("keras_model.h5")
LABELS = open("labels.txt").read().splitlines()

class2_start = None
last_seen = None
telegram_sent = False

# ===== TELEGRAM =====
def send_telegram(prob):
    msg = f"""⚠️ ALERT
eyes_closing detected
Confidence: {prob*100:.1f}%
Duration: 10s"""
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )

# ===== RECEIVE IMAGE =====
@app.route("/upload", methods=["POST"])
def upload():
    global class2_start, last_seen, telegram_sent

    file = request.files["image"]
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = MODEL.predict(img)[0]
    now = time.time()

    detected = False
    prob = 0

    for i, p in enumerate(pred):
        if LABELS[i] == "eyes_close" and p >= 0.6:
            detected = True
            prob = p

    if detected:
        if not class2_start:
            class2_start = now
        last_seen = now

    if class2_start:
        if last_seen and now - last_seen < 2:
            if now - class2_start >= 10 and not telegram_sent:
                send_telegram(prob)
                telegram_sent = True
        else:
            class2_start = None
            last_seen = None
            telegram_sent = False


    return "OK"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
