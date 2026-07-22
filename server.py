from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import time
import json
import os

# ✅ แก้ไขการ Import MediaPipe ให้รันบน Render ได้สมบูรณ์
import mediapipe as mp
from mediapipe.solutions import face_mesh as mp_face_mesh

app = Flask(__name__)

# Initialize MediaPipe FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Preset Data Structure
PRESETS_FILE = 'presets.json'
default_presets = {
    "1": {"name": "Driver 1", "ear_open": 0.30, "ear_closed": 0.12},
    "2": {"name": "Driver 2", "ear_open": 0.30, "ear_closed": 0.12},
    "3": {"name": "Driver 3", "ear_open": 0.30, "ear_closed": 0.12}
}

if os.path.exists(PRESETS_FILE):
    with open(PRESETS_FILE, 'r') as f:
        presets = json.load(f)
else:
    presets = default_presets

state = {
    "active_preset": "1",
    "eye_closure_pct": 0,
    "closed_duration": 0.0,
    "alarm_active": False,
    "last_closed_time": None,
    "latest_ear": 0.0,
    "calibrating": False
}

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

@app.route('/')
def index():
    return render_template_string(HTML_UI)

@app.route('/upload', methods=['POST'])
def upload():
    global state
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image"}), 400

    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Decode failed"}), 400

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_ear = 0.0
    closure_pct = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        current_ear = (left_ear + right_ear) / 2.0
        state["latest_ear"] = round(current_ear, 4)

        # Calculate Closure % based on active preset
        p_data = presets[state["active_preset"]]
        ear_open = p_data["ear_open"]
        ear_closed = p_data["ear_closed"]

        if ear_open > ear_closed:
            pct = ((ear_open - current_ear) / (ear_open - ear_closed)) * 100.0
            closure_pct = max(0.0, min(100.0, pct))

    state["eye_closure_pct"] = round(closure_pct, 1)

    # Drowsiness Logic (Eyes Closed >= 70% for > 10 Seconds)
    now = time.time()
    if closure_pct >= 70.0:
        if state["last_closed_time"] is None:
            state["last_closed_time"] = now
            state["closed_duration"] = 0.0
        else:
            state["closed_duration"] = round(now - state["last_closed_time"], 1)

        if state["closed_duration"] >= 10.0:
            state["alarm_active"] = True
    else:
        state["last_closed_time"] = None
        state["closed_duration"] = 0.0
        state["alarm_active"] = False

    return jsonify({
        "alert": state["alarm_active"],
        "closure_pct": state["eye_closure_pct"],
        "duration": state["closed_duration"],
        "ear": state["latest_ear"]
    })

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({"state": state, "presets": presets})

@app.route('/api/select_preset', methods=['POST'])
def select_preset():
    p_id = request.json.get('preset_id')
    if p_id in presets:
        state["active_preset"] = p_id
        return jsonify({"success": True})
    return jsonify({"error": "Invalid preset"}), 400

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    # step: 'open' or 'closed'
    p_id = state["active_preset"]
    step = request.json.get('step')
    if step == 'open':
        presets[p_id]["ear_open"] = state["latest_ear"]
    elif step == 'closed':
        presets[p_id]["ear_closed"] = state["latest_ear"]

    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f)
    return jsonify({"success": True, "preset": presets[p_id]})

HTML_UI = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Drowsiness Monitor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
        .card { background: #1e1e1e; border-radius: 12px; padding: 20px; margin: 15px auto; max-width: 500px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .btn { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 5px; font-weight: bold; }
        .btn-primary { background: #007bff; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-active { border: 2px solid #fff; box-shadow: 0 0 10px #007bff; }
        .stat-box { font-size: 2rem; font-weight: bold; margin: 10px 0; }
        .alert-active { background: #ff0000 !important; animation: blink 0.5s infinite alternate; }
        @keyframes blink { from { opacity: 1; } to { opacity: 0.5; } }
    </style>
</head>
<body>
    <h1>🚗 Driver Drowsiness Monitor</h1>

    <div class="card" id="alertCard">
        <h2>STATUS: <span id="statusText">NORMAL</span></h2>
        <div class="stat-box">ตาปิด: <span id="pctText">0</span>%</div>
        <div>หลับตาต่อเนื่อง: <span id="durText">0</span> / 10 วินาที</div>
        <div>Current EAR: <span id="earText">0.00</span></div>
    </div>

    <div class="card">
        <h3>เลือก Driver Preset</h3>
        <button class="btn btn-primary" id="p1" onclick="setPreset('1')">Preset 1</button>
        <button class="btn btn-primary" id="p2" onclick="setPreset('2')">Preset 2</button>
        <button class="btn btn-primary" id="p3" onclick="setPreset('3')">Preset 3</button>
    </div>

    <div class="card">
        <h3>🔧 Calibrate ตาล่าสุด (Preset ปัจจุบัน)</h3>
        <p>1. นำหน้าเข้าใกล้กล้อง เปิดตาปกติ แล้วกด "บันทึกตอนลืมตา"</p>
        <button class="btn btn-success" onclick="calibrate('open')">1. บันทึกตอนลืมตา</button>
        <p>2. ลองหลับตา แล้วกด "บันทึกตอนหลับตา"</p>
        <button class="btn btn-danger" onclick="calibrate('closed')">2. บันทึกตอนหลับตา</button>
        <div style="margin-top:10px; font-size:0.9rem; color:#aaa;" id="presetInfo"></div>
    </div>

    <script>
        async function updateData() {
            try {
                let res = await fetch('/api/state');
                let data = await res.json();
                let st = data.state;
                let ps = data.presets[st.active_preset];

                document.getElementById('pctText').innerText = st.eye_closure_pct;
                document.getElementById('durText').innerText = st.closed_duration;
                document.getElementById('earText').innerText = st.latest_ear;

                // Preset Active Styling
                ['1','2','3'].forEach(id => {
                    let btn = document.getElementById('p'+id);
                    if(id === st.active_preset) btn.classList.add('btn-active');
                    else btn.classList.remove('btn-active');
                });

                document.getElementById('presetInfo').innerText = 
                    `Active: ${ps.name} | EAR Open: ${ps.ear_open} | EAR Closed: ${ps.ear_closed}`;

                // Alarm Alert UI
                let card = document.getElementById('alertCard');
                let stText = document.getElementById('statusText');
                if(st.alarm_active) {
                    card.classList.add('alert-active');
                    stText.innerText = "⚠️ SLEEPING DETECTED!";
                } else {
                    card.classList.remove('alert-active');
                    stText.innerText = "NORMAL";
                }
            } catch(e) {}
        }

        async function setPreset(id) {
            await fetch('/api/select_preset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({preset_id: id})
            });
            updateData();
        }

        async function calibrate(step) {
            await fetch('/api/calibrate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({step: step})
            });
            alert('Calibrated step: ' + step);
            updateData();
        }

        setInterval(updateData, 500);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
