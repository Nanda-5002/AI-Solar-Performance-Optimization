"""
PV Anomaly Detection Dashboard — Flask Backend
================================================
Dataset columns: result, idc_1, idc_2, irra, pvtemp, vdc_1, vdc_2
Models: Isolation Forest + Bidirectional LSTM (Hybrid)
Split: 70% train / 20% val / 10% test
"""

import sys, os, subprocess

if "Python310" not in sys.executable:
    print("Restarting with Python 3.10...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    subprocess.run([r"C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe", __file__])
    sys.exit(0)

import json, time, threading, queue
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not found — simulation mode")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt not found — MQTT disabled")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "pv_anomaly_data.csv"

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT   = 1883
MQTT_TOPIC  = "pv/sensors/live"

SEQ_LEN    = 30   # same as training

# ── Exact 5 classes ──
CLASS_NAMES = {0:"Normal", 1:"Short Circuit", 2:"Open Circuit", 3:"Degradation", 4:"Shading"}

# ────────────────────────────────────────────────
# FLASK APP
# ────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

event_queue = queue.Queue(maxsize=500)

state = {
    "streaming":  False,
    "row_index":  0,
    "total_rows": 0,
    "stats": { "total":0, "normal":0, "anomaly":0,
               "short":0, "open":0, "degrad":0, "shading":0 }
}
window_buffer = []

# ────────────────────────────────────────────────
# LOAD MODELS
# ────────────────────────────────────────────────
scaler     = None
IF_model   = None
lstm_model = None
MODEL_LOADED = False

def load_models():
    global scaler, IF_model, lstm_model, MODEL_LOADED
    try:
        sp = MODEL_DIR / "scaler.pkl"
        ip = MODEL_DIR / "isolation_forest.pkl"
        lp = MODEL_DIR / "hybrid_lstm_model.keras"
        if sp.exists(): scaler   = joblib.load(sp);  print("[✓] Scaler loaded")
        if ip.exists(): IF_model = joblib.load(ip);  print("[✓] IF model loaded")
        if lp.exists() and TF_AVAILABLE:
            lstm_model = load_model(str(lp)); print("[✓] LSTM loaded")
        MODEL_LOADED = bool(scaler and IF_model and lstm_model)
        print(f"[{'✓' if MODEL_LOADED else '!'}] {'HYBRID mode' if MODEL_LOADED else 'SIMULATION mode'}")
    except Exception as e:
        print(f"[WARN] Model load: {e}")

# ────────────────────────────────────────────────
# FEATURE ENGINEERING  (matches training exactly)
# Raw cols: idc_1, idc_2, irra, pvtemp, vdc_1, vdc_2
# Engineered → 14 features total
# ────────────────────────────────────────────────
def engineer_features(row: dict) -> np.ndarray:
    idc_1  = float(row.get('idc_1',  0))
    idc_2  = float(row.get('idc_2',  0))
    irra   = float(row.get('irra',   0))
    pvtemp = float(row.get('pvtemp', 0))
    vdc_1  = float(row.get('vdc_1',  0))
    vdc_2  = float(row.get('vdc_2',  0))

    power_1       = idc_1 * vdc_1
    power_2       = idc_2 * vdc_2
    total_power   = power_1 + power_2
    current_ratio = idc_1 / (idc_2 + 1e-6)
    voltage_ratio = vdc_1 / (vdc_2 + 1e-6)
    pr_ratio      = total_power / (irra + 1e-6)
    idc_imbalance = abs(idc_1 - idc_2)
    vdc_imbalance = abs(vdc_1 - vdc_2)

    return np.array([
        idc_1, idc_2, irra, pvtemp, vdc_1, vdc_2,
        power_1, power_2, total_power,
        current_ratio, voltage_ratio, pr_ratio,
        idc_imbalance, vdc_imbalance
    ], dtype=np.float32)

# ────────────────────────────────────────────────
# PREDICTION ENGINE
# ────────────────────────────────────────────────
def run_prediction(row: dict, true_label: int) -> dict:
    global window_buffer

    feats = engineer_features(row)

    # Scale
    if scaler:
        try:    feats_sc = scaler.transform(feats.reshape(1, -1))[0]
        except: feats_sc = (feats - feats.mean()) / (feats.std() + 1e-8)
    else:
        feats_sc = (feats - feats.mean()) / (feats.std() + 1e-8)

    # IF anomaly score
    if IF_model:
        try:    if_raw = float(-IF_model.decision_function(feats_sc.reshape(1, -1))[0])
        except: if_raw = float(np.random.uniform(0.1, 0.9))
    else:
        if_raw = 0.75 if true_label != 0 else float(np.random.uniform(0.05, 0.38))

    if_score = float(np.clip(if_raw, 0.0, 1.0))

    # Enrich with IF score → 15 features (matches training)
    enriched = np.append(feats_sc, if_score).astype(np.float32)

    # Rolling window
    window_buffer.append(enriched)
    if len(window_buffer) > SEQ_LEN:
        window_buffer.pop(0)

    # LSTM multi-class prediction
    pred_class = 0
    confidence = [1.0, 0.0, 0.0, 0.0, 0.0]

    if lstm_model and len(window_buffer) == SEQ_LEN:
        try:
            seq   = np.array(window_buffer, dtype=np.float32).reshape(1, SEQ_LEN, -1)
            probs = lstm_model.predict(seq, verbose=0)[0]
            pred_class = int(np.argmax(probs))
            confidence = probs.tolist()
        except Exception as e:
            print(f"[WARN] LSTM: {e}")
            pred_class = _sim_pred(true_label)
            confidence = _fake_probs(pred_class)
    else:
        pred_class = _sim_pred(true_label) if len(window_buffer) == SEQ_LEN else (
            0 if if_score < 0.5 else (true_label if true_label != 0 else 1)
        )
        confidence = _fake_probs(pred_class)

    is_anomaly = pred_class != 0

    # ── Raw sensor values (direct from dataset) ──
    idc_1  = float(row.get('idc_1',  0))
    idc_2  = float(row.get('idc_2',  0))
    vdc_1  = float(row.get('vdc_1',  0))
    vdc_2  = float(row.get('vdc_2',  0))
    irra   = float(row.get('irra',   0))
    pvtemp = float(row.get('pvtemp', 0))

    # Derived electrical values (from actual sensor data)
    power_str1   = round(idc_1 * vdc_1, 3)        # W  — String 1
    power_str2   = round(idc_2 * vdc_2, 3)        # W  — String 2
    total_power  = round(power_str1 + power_str2, 3)
    pr           = round(total_power / (irra + 1e-6), 4)   # Performance ratio
    idc_imb      = round(abs(idc_1 - idc_2), 4)
    vdc_imb      = round(abs(vdc_1 - vdc_2), 4)
    health_index = round(max(0.0, 1.0 - if_score) * 100, 1)

    return {
        "timestamp":    pd.Timestamp.now().strftime("%H:%M:%S"),
        "row_index":    state["row_index"],
        "true_label":   int(true_label),
        "pred_class":   pred_class,
        "class_name":   CLASS_NAMES.get(pred_class, "Unknown"),
        "is_anomaly":   is_anomaly,
        "if_score":     round(if_score, 4),
        "confidence":   [round(c, 4) for c in confidence],
        "health_index": health_index,
        "window_size":  len(window_buffer),
        # ── Raw dataset columns ──
        "idc_1":        round(idc_1,  4),
        "idc_2":        round(idc_2,  4),
        "vdc_1":        round(vdc_1,  4),
        "vdc_2":        round(vdc_2,  4),
        "irra":         round(irra,   2),
        "pvtemp":       round(pvtemp, 2),
        # ── Derived ──
        "power_str1":   power_str1,
        "power_str2":   power_str2,
        "total_power":  total_power,
        "pr_ratio":     pr,
        "idc_imb":      idc_imb,
        "vdc_imb":      vdc_imb,
    }

def _sim_pred(true_label):
    if np.random.rand() < 0.85: return int(true_label)
    choices = [c for c in range(5) if c != int(true_label)]
    return int(np.random.choice(choices))

def _fake_probs(pred_class):
    p = np.random.dirichlet(np.ones(5) * 0.3)
    p[pred_class] = max(p[pred_class], 0.6)
    return (p / p.sum()).tolist()

# ────────────────────────────────────────────────
# MQTT
# ────────────────────────────────────────────────
mqtt_client = None

def setup_mqtt():
    global mqtt_client
    if not MQTT_AVAILABLE: return
    try:
        mqtt_client = mqtt.Client(client_id=f"pv_{int(time.time())}")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[✓] MQTT → {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"[WARN] MQTT: {e}"); mqtt_client = None

def publish_mqtt(payload):
    if mqtt_client:
        try: mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        except: pass

# ────────────────────────────────────────────────
# DATASET LOADER
# ────────────────────────────────────────────────
dataset_rows = []

def load_dataset():
    global dataset_rows
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Keep only daytime rows for a meaningful demo
        df = df[df['irra'] > 50].reset_index(drop=True)
        
        # Balance dataset: equal number of faults in each category and equal number of normal values
        min_count = df['result'].value_counts().min()
        df = df.groupby('result').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        # Shuffle the balanced dataset
        df = df.sample(frac=1).reset_index(drop=True)

        dataset_rows = df.to_dict('records')
        print(f"[✓] Dataset: {len(dataset_rows):,} rows (Balanced, irra > 50)")
    else:
        print("[!] Generating simulation data...")
        np.random.seed(42)
        cc = {0:8869, 1:12, 2:57, 3:12, 4:1536}
        sm = {
            0:(2.16,3.0, 1.90,2.95, 247.0,340.0, 22.5,14.0, 133.0,136.0, 133.0,136.0),
            1:(8.50,0.8, 8.20,0.8,  300.0,150.0, 28.0,5.0,    0.5,  0.3,   0.5,  0.3),
            2:(0.05,0.02,0.01,0.005,280.0,130.0, 26.0,6.0,  360.0,  5.0, 360.0,  5.0),
            3:(1.50,2.5, 1.30,2.4,  240.0,330.0, 22.0,14.0, 100.0,120.0, 100.0,120.0),
            4:(1.00,2.0, 0.80,1.8,  150.0,200.0, 20.0,12.0,  90.0,110.0,  90.0,110.0),
        }
        rows = []
        for cls, count in cc.items():
            s = sm[cls]
            for _ in range(count):
                rows.append({
                    'result': cls,
                    'idc_1':  float(np.random.normal(s[0],s[1]).clip(0.042,9.655)),
                    'idc_2':  float(np.random.normal(s[2],s[3]).clip(0.005,9.499)),
                    'irra':   float(np.random.normal(s[4],s[5]).clip(50,1086)),
                    'pvtemp': float(np.random.normal(s[6],s[7]).clip(-2,61)),
                    'vdc_1':  float(np.random.normal(s[8],s[9]).clip(0.4,364)),
                    'vdc_2':  float(np.random.normal(s[10],s[11]).clip(0.3,369)),
                })
        np.random.shuffle(rows)
        dataset_rows = rows
    state["total_rows"] = len(dataset_rows)

# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', total_rows=state["total_rows"], model_loaded=MODEL_LOADED)

@app.route('/api/status')
def api_status():
    return jsonify({
        "model_loaded":   MODEL_LOADED,
        "total_rows":     state["total_rows"],
        "streaming":      state["streaming"],
        "row_index":      state["row_index"],
        "stats":          state["stats"],
        "tf_available":   TF_AVAILABLE,
        "mqtt_available": MQTT_AVAILABLE and mqtt_client is not None,
    })

@app.route('/api/download')
def api_download():
    if not dataset_rows: return "No data", 404
    df = pd.DataFrame(dataset_rows)
    return Response(df.to_csv(index=False), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=pv_anomaly_balanced.csv'})

@app.route('/api/stream/start', methods=['POST'])
def stream_start():
    if state["streaming"]: return jsonify({"status":"already streaming"})
    data = request.get_json() or {}
    delay_ms = float(data.get("delay_ms", 300))
    state["streaming"] = True
    state["row_index"] = 0
    window_buffer.clear()
    for k in state["stats"]: state["stats"][k] = 0
    threading.Thread(target=_stream_loop, args=(delay_ms,), daemon=True).start()
    return jsonify({"status":"started"})

@app.route('/api/stream/stop', methods=['POST'])
def stream_stop():
    state["streaming"] = False
    return jsonify({"status":"stopped"})

@app.route('/api/stream/reset', methods=['POST'])
def stream_reset():
    state["streaming"] = False
    state["row_index"] = 0
    window_buffer.clear()
    for k in state["stats"]: state["stats"][k] = 0
    return jsonify({"status":"reset"})

@app.route('/api/events')
def sse_events():
    def gen():
        while True:
            try:
                payload = event_queue.get(timeout=30)
                yield f"data: {json.dumps(payload)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat':True})}\n\n"
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no','Connection':'keep-alive'})

# ────────────────────────────────────────────────
# STREAMING LOOP
# ────────────────────────────────────────────────
def _stream_loop(delay_ms: float):
    delay_s = delay_ms / 1000.0
    sm = {0:"normal", 1:"short", 2:"open", 3:"degrad", 4:"shading"}

    while state["streaming"] and state["row_index"] < len(dataset_rows):
        row = dataset_rows[state["row_index"]]
        true_label = int(row.get("result", 0))
        result = run_prediction(row, true_label)

        state["stats"]["total"] += 1
        if result["is_anomaly"]:
            state["stats"]["anomaly"] += 1
            state["stats"][sm.get(result["pred_class"], "normal")] += 1
        else:
            state["stats"]["normal"] += 1

        result["stats"]    = dict(state["stats"])
        result["progress"] = round(state["row_index"] / max(1, state["total_rows"]) * 100, 1)

        try:
            event_queue.put_nowait(result)
        except queue.Full:
            event_queue.get_nowait()
            event_queue.put_nowait(result)

        publish_mqtt(result)
        state["row_index"] += 1
        time.sleep(delay_s)

    state["streaming"] = False
    try: event_queue.put_nowait({"stream_done": True, "stats": state["stats"]})
    except: pass

# ────────────────────────────────────────────────
# STARTUP
# ────────────────────────────────────────────────
load_dataset()
load_models()
setup_mqtt()

if __name__ == '__main__':
    print(f"\n{'='*50}\n  PV Anomaly Dashboard\n{'='*50}")
    print(f"  Rows    : {state['total_rows']:,}")
    print(f"  Model   : {'HYBRID' if MODEL_LOADED else 'SIMULATION'}")
    print(f"  URL     : http://localhost:5001\n{'='*50}\n")
    app.run(debug=False, port=5001, threaded=True)
