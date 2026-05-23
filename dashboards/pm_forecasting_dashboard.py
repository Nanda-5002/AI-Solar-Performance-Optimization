"""
Solar PM + Power Forecasting Dashboard  v3
Per-inverter mode: 22 inverter dropdown, per-inverter models,
in-process MQTT simulation, white theme, speed/start/stop, notifications.
"""

import os, json, time, threading, queue, math
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template_string, request, jsonify

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
GEN_CSV = os.path.join(BASE, "Plant_1_Generation_Data (1).csv")
WEA_CSV = os.path.join(BASE, "Plant_1_Weather_Sensor_Data.csv")

MAX_ROWS     = 2000
stream_delay = {"value": 0.8}

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  — per-inverter rows (no aggregation)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data():
    gen     = pd.read_csv(GEN_CSV)
    weather = pd.read_csv(WEA_CSV)
    gen.columns = gen.columns.str.strip()
    weather.columns = weather.columns.str.strip()

    gen['DATE_TIME']     = pd.to_datetime(gen['DATE_TIME'], dayfirst=True, errors='coerce')
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], errors='coerce')
    
    # Drop rows with unparseable dates
    gen     = gen.dropna(subset=['DATE_TIME'])
    weather = weather.dropna(subset=['DATE_TIME'])

    # Round to nearest 15 mins to ensure merge works even with small offsets
    gen['DATE_TIME']     = gen['DATE_TIME'].dt.round('15min')
    weather['DATE_TIME'] = weather['DATE_TIME'].dt.round('15min')

    # Keep only needed gen columns to avoid PLANT_ID conflicts on merge
    available_gen = [c for c in gen.columns if c.upper() == 'SOURCE_KEY']
    sk_col = available_gen[0] if available_gen else 'SOURCE_KEY'

    gen_cols = ['DATE_TIME', sk_col, 'DC_POWER', 'AC_POWER']
    gen = gen[gen_cols]
    if sk_col != 'SOURCE_KEY': gen.rename(columns={sk_col: 'SOURCE_KEY'}, inplace=True)

    # Keep per-inverter rows — merge weather (plant-level sensors)
    # Most likely weather doesn't have SOURCE_KEY per inverter in this dataset, but check anyway
    if 'SOURCE_KEY' in weather.columns and weather['SOURCE_KEY'].nunique() > 1:
        df = gen.merge(weather, on=['DATE_TIME', 'SOURCE_KEY'], how='inner')
    else:
        # If weather is plant-level, drop its SOURCE_KEY before merge if it exists
        if 'SOURCE_KEY' in weather.columns: weather = weather.drop(columns=['SOURCE_KEY'])
        df = gen.merge(weather, on='DATE_TIME', how='inner')
    
    # Filter daytime only
    df = df[df['IRRADIATION'] > 0.001].reset_index(drop=True)
    
    if len(df) == 0:
        print("[Error] No overlapping daytime data found between Generation and Weather CSVs!")
        print(f"Gen range: {gen['DATE_TIME'].min()} to {gen['DATE_TIME'].max()}")
        print(f"Weather range: {weather['DATE_TIME'].min()} to {weather['DATE_TIME'].max()}")
        raise ValueError("Empty dataset after merge correctly")

    n_inv = df['SOURCE_KEY'].nunique()
    inv_list = sorted(df['SOURCE_KEY'].unique().tolist())
    print(f"[Data] {n_inv} inverters, {len(df)} daytime rows total")
    return df, inv_list

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (per-inverter lag features)
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_FEATURES = [
    'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
    'POWER_LAG_1', 'POWER_LAG_2', 'POWER_LAG_3',
    'POWER_LAG_4', 'POWER_LAG_5', 'HOUR_SIN', 'HOUR_COS'
]
PM_FEATURES = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
               'TEMP_DELTA', 'DC_AC_RATIO']

def add_features(df):
    if len(df) == 0: return df
    df = df.copy().sort_values(['SOURCE_KEY', 'DATE_TIME'])
    # Lag features WITHIN each inverter's timeline
    for lag in range(1, 6):
        df[f'POWER_LAG_{lag}'] = df.groupby('SOURCE_KEY')['AC_POWER'].shift(lag)
    df['HOUR']      = df['DATE_TIME'].dt.hour
    df['HOUR_SIN']  = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS']  = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['TEMP_DELTA'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
    df['DC_AC_RATIO'] = df['DC_POWER'] / (df['AC_POWER'] + 1e-6)
    
    # Fill lags for first few rows of each inverter instead of dropping everything
    df = df.ffill().bfill() 
    
    df = df.dropna(subset=FORECAST_FEATURES + PM_FEATURES).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_forecast_model(df):
    split = int(0.8 * len(df))
    X_tr, y_tr = df[FORECAST_FEATURES][:split], df['AC_POWER'][:split]
    X_te, y_te = df[FORECAST_FEATURES][split:], df['AC_POWER'][split:]
    model = RandomForestRegressor(n_estimators=120, max_depth=12,
                                  min_samples_split=8, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = math.sqrt(np.mean((y_te.values - pred) ** 2))
    r2   = r2_score(y_te, pred)
    print(f"[Forecast] RMSE={rmse:.1f}  R²={r2:.4f}  (per-inverter rows)")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# PM MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_pm_model(df):
    df2 = df.copy()
    # Labels per-inverter: efficiency ~ AC/DC (should be ~0.92-0.97)
    eff = df2['AC_POWER'] / (df2['DC_POWER'] + 1e-6)
    eff_low     = eff < eff.median() * 0.95
    temp_stress = df2['MODULE_TEMPERATURE'] > 52
    dc_ac_high  = df2['DC_AC_RATIO'] > df2['DC_AC_RATIO'].quantile(0.80)
    df2['LABEL'] = (eff_low | temp_stress | dc_ac_high).astype(int)

    np.random.seed(42)
    noise = np.random.choice(df2.index, size=int(0.05 * len(df2)), replace=False)
    df2.loc[noise, 'LABEL'] = 1 - df2.loc[noise, 'LABEL']

    X, y = df2[PM_FEATURES], df2['LABEL']
    # No stratify — avoids crash when minority class has too few samples
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=80, max_depth=6,
                                   min_samples_split=20, min_samples_leaf=10,
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    acc = (model.predict(X_te) == y_te).mean()
    print(f"[PM Model] Accuracy={acc:.3f}  DC/AC ratio per inverter: "
          f"{df2['DC_AC_RATIO'].median():.2f} (physically meaningful)")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Loading data & training per-inverter models …")
raw_df, inv_list = prepare_data()
feat_df = add_features(raw_df)

fc_model = build_forecast_model(feat_df)
pm_model = build_pm_model(feat_df)

# Default active inverter = first one
active_inverter = {"key": inv_list[0]}

def get_inverter_stream(key):
    df = feat_df[feat_df['SOURCE_KEY'] == key].sort_values('DATE_TIME')
    return df.iloc[:MAX_ROWS].reset_index(drop=True)

current_stream_df = {"df": get_inverter_stream(inv_list[0])}
print(f"[Stream] Default inverter: {inv_list[0]}  rows: {len(current_stream_df['df'])}")
print(f"[Stream] All {len(inv_list)} inverters available in dropdown")
print("="*60)
print("\n✅ Open http://127.0.0.1:5000 — select an inverter and click ▶ Start\n")

# ─────────────────────────────────────────────────────────────────────────────
# MQTT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
mqtt_queue   = queue.Queue(maxsize=300)
stream_active = {"running": False}

def mqtt_publisher():
    stream_active["running"] = True
    df = current_stream_df["df"]
    for _, row in df.iterrows():
        if not stream_active["running"]:
            break
        payload = row.to_dict()
        for k, v in payload.items():
            if hasattr(v, 'item'):   payload[k] = v.item()
            elif isinstance(v, float) and math.isnan(v): payload[k] = 0.0
        if hasattr(payload.get('DATE_TIME'), 'isoformat'):
            payload['DATE_TIME'] = payload['DATE_TIME'].isoformat()
        try:    mqtt_queue.put_nowait(payload)
        except: pass
        time.sleep(stream_delay["value"])
    stream_active["running"] = False

def start_publisher():
    while not mqtt_queue.empty():
        try: mqtt_queue.get_nowait()
        except: break
    t = threading.Thread(target=mqtt_publisher, daemon=True)
    t.start()

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def infer(row: dict) -> dict:
    fc_in = pd.DataFrame([{f: row.get(f, 0) for f in FORECAST_FEATURES}])
    pm_in = pd.DataFrame([{f: row.get(f, 0) for f in PM_FEATURES}])

    forecast_kw = float(fc_model.predict(fc_in)[0])
    maint_prob  = float(pm_model.predict_proba(pm_in)[0, 1])
    maint_flag  = int(pm_model.predict(pm_in)[0])

    actual_kw   = float(row.get('AC_POWER', 0))
    dc_power    = float(row.get('DC_POWER', 0))
    irradiation = float(row.get('IRRADIATION', 0))
    amb_temp    = float(row.get('AMBIENT_TEMPERATURE', 0))
    mod_temp    = float(row.get('MODULE_TEMPERATURE', 0))
    temp_delta  = float(row.get('TEMP_DELTA', mod_temp - amb_temp))
    dc_ac_ratio = float(row.get('DC_AC_RATIO', dc_power / (actual_kw + 1e-6)))
    # Per-inverter efficiency: AC/DC  (~0.92–0.97 for healthy inverter)
    efficiency  = round((actual_kw / (dc_power + 1e-6)) * 100, 1)

    health_index    = round(max(0, min(100, 100 * (1 - maint_prob))), 1)
    perf_ratio      = round(min(150, (actual_kw / (forecast_kw + 1e-6)) * 100), 1)
    power_loss      = round(max(0, forecast_kw - actual_kw), 2)
    power_deviation = round(((actual_kw - forecast_kw) / (forecast_kw + 1e-6)) * 100, 1)

    if mod_temp >= 55:   temp_stress = "CRITICAL"
    elif mod_temp >= 50: temp_stress = "HIGH"
    elif mod_temp >= 45: temp_stress = "MODERATE"
    else:                temp_stress = "NORMAL"

    if maint_prob >= 0.75:   risk_level = "CRITICAL"
    elif maint_prob >= 0.50: risk_level = "HIGH"
    elif maint_prob >= 0.30: risk_level = "MODERATE"
    else:                    risk_level = "LOW"

    if health_index >= 80:   health_status = "HEALTHY"
    elif health_index >= 60: health_status = "DEGRADED"
    elif health_index >= 40: health_status = "WARNING"
    else:                    health_status = "CRITICAL"

    return {
        "timestamp":       row.get('DATE_TIME', datetime.now().isoformat()),
        "inverter":        row.get('SOURCE_KEY', active_inverter["key"]),
        "health_index":    health_index,
        "health_status":   health_status,
        "maint_required":  bool(maint_flag),
        "maint_prob":      round(maint_prob * 100, 1),
        "risk_level":      risk_level,
        "forecast_kw":     round(forecast_kw, 2),
        "actual_kw":       round(actual_kw, 2),
        "dc_power":        round(dc_power, 2),
        "dc_ac_ratio":     round(dc_ac_ratio, 3),
        "efficiency_pct":  efficiency,
        "perf_ratio":      perf_ratio,
        "power_loss_kw":   power_loss,
        "power_deviation": power_deviation,
        "irradiation":     round(irradiation, 4),
        "amb_temp":        round(amb_temp, 2),
        "mod_temp":        round(mod_temp, 2),
        "temp_delta":      round(temp_delta, 2),
        "temp_stress":     temp_stress,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/inverters')
def get_inverters():
    return jsonify({"inverters": inv_list, "active": active_inverter["key"]})

@app.route('/select/<key>')
def select_inverter(key):
    if key not in inv_list:
        return jsonify({"error": "unknown inverter"}), 400
    stream_active["running"] = False
    time.sleep(0.3)
    active_inverter["key"] = key
    current_stream_df["df"] = get_inverter_stream(key)
    return jsonify({"selected": key, "rows": len(current_stream_df["df"])})

@app.route('/start')
def start_stream():
    if not stream_active["running"]:
        start_publisher()
        return jsonify({"status": "started", "inverter": active_inverter["key"]})
    return jsonify({"status": "already_running"})

@app.route('/stop')
def stop_stream():
    stream_active["running"] = False
    return jsonify({"status": "stopped"})

@app.route('/speed/<float:secs>')
def set_speed(secs):
    stream_delay["value"] = max(0.1, min(5.0, secs))
    return jsonify({"delay": stream_delay["value"]})

@app.route('/stream')
def stream():
    def gen():
        while True:
            try:
                row    = mqtt_queue.get(timeout=5)
                result = infer(row)
                yield f"data: {json.dumps(result)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'keepalive': True})}\n\n"
    return Response(gen(), mimetype='text/event-stream',
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ─────────────────────────────────────────────────────────────────────────────
# HTML (white theme, inverter dropdown)
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Solar Intelligence Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f1f5f9; --surface:#fff; --surface2:#f8fafc;
  --border:#e2e8f0; --border2:#cbd5e1;
  --t1:#0f172a; --t2:#475569; --t3:#94a3b8;
  --blue:#3b82f6; --green:#16a34a; --red:#dc2626;
  --orange:#ea580c; --yellow:#d97706; --purple:#7c3aed;
  --sh:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  --sh2:0 4px 6px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.06);
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t1);min-height:100vh;}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;
  padding:12px 26px;background:var(--surface);border-bottom:1px solid var(--border);
  box-shadow:var(--sh);position:sticky;top:0;z-index:100;}
.hdr-l{display:flex;align-items:center;gap:11px;}
.logo{width:38px;height:38px;border-radius:9px;background:linear-gradient(135deg,#3b82f6,#06b6d4);
  display:flex;align-items:center;justify-content:center;font-size:19px;box-shadow:0 2px 8px rgba(59,130,246,.3);}
.hdr-t h1{font-size:16px;font-weight:700;} .hdr-t p{font-size:10px;color:var(--t2);margin-top:1px;}
.hdr-r{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}

/* Inverter selector */
.inv-wrap{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--t2);}
.inv-select{
  border:1.5px solid var(--border2);border-radius:8px;padding:5px 10px;
  font-size:12px;font-weight:600;color:var(--t1);background:#fff;cursor:pointer;
  max-width:200px;
}
.inv-select:focus{outline:none;border-color:var(--blue);}

/* Status pill */
.pill{display:flex;align-items:center;gap:6px;background:var(--surface2);
  border:1px solid var(--border);border-radius:20px;padding:4px 11px;font-size:11px;color:var(--t2);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--t3);}
.dot.on{background:var(--green);animation:pulse 1.4s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Speed */
.spd{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--t2);}
.spd input{width:85px;accent-color:var(--blue);cursor:pointer;}
#spd-val{font-weight:700;color:var(--blue);min-width:30px;}

/* Buttons */
.btn{border:none;border-radius:8px;padding:6px 14px;font-size:12px;font-weight:600;
  cursor:pointer;display:flex;align-items:center;gap:5px;transition:transform .12s;}
.btn:hover{transform:translateY(-1px);}
.btn-start{background:var(--blue);color:#fff;box-shadow:0 2px 8px rgba(59,130,246,.3);}
.btn-stop{background:#fff;color:var(--red);border:1.5px solid var(--red);}

/* Main */
.main{padding:18px 26px 64px;}

/* KPI */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:11px;margin-bottom:14px;}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:11px;
  padding:14px;position:relative;overflow:hidden;box-shadow:var(--sh);transition:transform .18s;}
.kpi:hover{transform:translateY(-2px);}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--ka,#3b82f6);border-radius:11px 11px 0 0;}
.kpi-lbl{font-size:9px;font-weight:700;letter-spacing:.8px;color:var(--t2);text-transform:uppercase;margin-bottom:5px;}
.kpi-val{font-size:22px;font-weight:800;color:var(--t1);line-height:1;}
.kpi-unit{font-size:10px;color:var(--t3);margin-top:3px;}
.kpi-badge{display:inline-flex;align-items:center;gap:3px;font-size:9px;font-weight:700;
  border-radius:5px;padding:2px 6px;margin-top:4px;}

/* Middle */
.mid{display:grid;grid-template-columns:260px 1fr 1fr;gap:12px;margin-bottom:14px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:11px;
  padding:18px;box-shadow:var(--sh);}
.card-t{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;
  color:var(--t2);margin-bottom:13px;}
.hlth-card{display:flex;flex-direction:column;align-items:center;text-align:center;}
.gscore{font-size:38px;font-weight:800;margin-top:6px;}
.glabel{font-size:11px;font-weight:700;letter-spacing:1.2px;margin-top:2px;color:var(--t2);}
.mrow{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid var(--border);}
.mrow:last-child{border-bottom:none;}
.mname{font-size:11px;color:var(--t2);} .mval{font-size:12px;font-weight:700;color:var(--t1);}
.rbadge{font-size:11px;font-weight:700;border-radius:7px;padding:5px 13px;
  letter-spacing:.5px;margin-bottom:11px;display:inline-block;}
.inv-badge{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
  border-radius:6px;padding:3px 9px;font-size:10px;font-weight:700;
  display:inline-block;margin-bottom:10px;}

/* Charts */
.charts{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:14px;}
.cwrap{position:relative;height:185px;}

/* Bottom */
.bot{display:grid;grid-template-columns:1fr 300px;gap:12px;}
.alist{display:flex;flex-direction:column;gap:6px;max-height:230px;overflow-y:auto;}
.alist::-webkit-scrollbar{width:4px;}
.alist::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}
.aitem{display:flex;gap:9px;background:#fef2f2;border:1px solid #fecaca;
  border-radius:7px;padding:8px 10px;animation:sli .25s ease;}
@keyframes sli{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.at{font-size:10px;font-weight:700;color:var(--red);}
.am{font-size:9px;color:var(--t2);margin-top:1px;}
.ab{font-size:10px;color:var(--t1);margin-top:2px;}
.no-a{font-size:11px;color:var(--t3);text-align:center;padding:18px 0;}

/* Footer */
.footer{position:fixed;bottom:0;left:0;right:0;background:var(--surface);
  border-top:1px solid var(--border);padding:5px 26px;
  display:flex;align-items:center;justify-content:space-between;
  font-size:10px;color:var(--t2);box-shadow:0 -2px 6px rgba(0,0,0,.04);}
.fstats{display:flex;gap:18px;}
.fstat{display:flex;gap:4px;align-items:center;}
.fstat span:first-child{color:var(--t3);}
.fstat span:last-child{font-weight:600;}

/* Toasts */
.tc{position:fixed;top:74px;right:18px;z-index:999;display:flex;flex-direction:column;gap:7px;}
.toast{background:#fff;border:1px solid var(--border);border-left:4px solid var(--red);
  border-radius:9px;padding:10px 14px;box-shadow:var(--sh2);max-width:290px;
  animation:tin .28s ease;}
@keyframes tin{from{opacity:0;transform:translateX(28px)}to{opacity:1;transform:none}}
.toast-t{font-size:11px;font-weight:700;color:var(--red);}
.toast-b{font-size:10px;color:var(--t2);margin-top:2px;}

@media(max-width:1100px){.mid{grid-template-columns:1fr 1fr;}.hlth-card{grid-column:1/-1;}
  .charts{grid-template-columns:1fr;}.bot{grid-template-columns:1fr;}}
@media(max-width:680px){.hdr{padding:8px 12px;}.main{padding:10px 12px;}.mid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="tc" id="tc"></div>

<header class="hdr">
  <div class="hdr-l">
    <div class="logo">☀️</div>
    <div class="hdr-t">
      <h1>Solar Intelligence Dashboard</h1>
      <p>Per-Inverter · Predictive Maintenance &amp; Power Forecasting · MQTT Live Stream</p>
    </div>
  </div>
  <div class="hdr-r">
    <div class="inv-wrap">
      <span>🔌 Inverter</span>
      <select class="inv-select" id="inv-select" onchange="selectInverter(this.value)">
        <option value="">Loading…</option>
      </select>
    </div>
    <div class="pill"><div class="dot" id="mdot"></div><span id="mstatus">Idle</span></div>
    <span id="rctr" style="font-size:11px;font-weight:600;color:var(--t2)">Row 0</span>
    <div class="spd">
      <span>Speed</span>
      <input type="range" id="spd" min="1" max="20" value="8" oninput="setSpeed(this.value)"/>
      <span id="spd-val">0.8s</span>
    </div>
    <button class="btn btn-start" id="btn-s" onclick="startStream()">▶ Start</button>
    <button class="btn btn-stop"  id="btn-x" onclick="stopStream()" style="display:none">⏹ Stop</button>
  </div>
</header>

<div class="main">
  <!-- KPI -->
  <div class="kpi-grid">
    <div class="kpi" style="--ka:#16a34a">
      <div class="kpi-lbl">Health Index</div>
      <div class="kpi-val" id="k-hi">—</div>
      <div class="kpi-unit">/ 100</div>
    </div>
    <div class="kpi" style="--ka:#dc2626">
      <div class="kpi-lbl">Maintenance Required</div>
      <div class="kpi-val" id="k-mf" style="font-size:18px">—</div>
      <div id="k-rb" class="kpi-badge">—</div>
    </div>
    <div class="kpi" style="--ka:#3b82f6">
      <div class="kpi-lbl">Forecasted AC Power</div>
      <div class="kpi-val" id="k-fc">—</div>
      <div class="kpi-unit">kW (single inverter)</div>
    </div>
    <div class="kpi" style="--ka:#0891b2">
      <div class="kpi-lbl">Actual AC Power</div>
      <div class="kpi-val" id="k-ac">—</div>
      <div class="kpi-unit">kW (single inverter)</div>
    </div>
    <div class="kpi" style="--ka:#7c3aed">
      <div class="kpi-lbl">Performance Ratio</div>
      <div class="kpi-val" id="k-pr">—</div>
      <div class="kpi-unit">%</div>
    </div>
    <div class="kpi" style="--ka:#16a34a">
      <div class="kpi-lbl">Inverter Efficiency</div>
      <div class="kpi-val" id="k-eff">—</div>
      <div class="kpi-unit">% (AC/DC)</div>
    </div>
    <div class="kpi" style="--ka:#ea580c">
      <div class="kpi-lbl">Power Loss</div>
      <div class="kpi-val" id="k-loss">—</div>
      <div class="kpi-unit">kW</div>
    </div>
    <div class="kpi" style="--ka:#d97706">
      <div class="kpi-lbl">Irradiation</div>
      <div class="kpi-val" id="k-irr">—</div>
      <div class="kpi-unit">W/m²</div>
    </div>
  </div>

  <!-- MIDDLE -->
  <div class="mid">
    <!-- Gauge -->
    <div class="card hlth-card">
      <div class="card-t">⚙ System Health Index</div>
      <div id="inv-badge-disp" class="inv-badge">Inverter: —</div>
      <svg style="width:180px;height:103px;overflow:visible" viewBox="0 0 200 110">
        <defs>
          <linearGradient id="gR"><stop offset="0%" stop-color="#dc2626"/><stop offset="100%" stop-color="#ea580c"/></linearGradient>
          <linearGradient id="gY"><stop offset="0%" stop-color="#d97706"/><stop offset="100%" stop-color="#ca8a04"/></linearGradient>
          <linearGradient id="gG"><stop offset="0%" stop-color="#16a34a"/><stop offset="100%" stop-color="#059669"/></linearGradient>
        </defs>
        <path d="M15,100 A85,85,0,0,1,185,100" fill="none" stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>
        <path d="M15,100 A85,85,0,0,1,64,22"   fill="none" stroke="url(#gR)" stroke-width="14" stroke-linecap="round" opacity=".2"/>
        <path d="M64,22 A85,85,0,0,1,136,22"   fill="none" stroke="url(#gY)" stroke-width="14" stroke-linecap="round" opacity=".2"/>
        <path d="M136,22 A85,85,0,0,1,185,100" fill="none" stroke="url(#gG)" stroke-width="14" stroke-linecap="round" opacity=".2"/>
        <path id="garc" d="M15,100 A85,85,0,0,1,15,100" fill="none" stroke="#16a34a" stroke-width="14" stroke-linecap="round"/>
        <line id="gneedle" x1="100" y1="100" x2="100" y2="22"
              stroke="#1e293b" stroke-width="2.5" stroke-linecap="round"
              transform="rotate(-90,100,100)"/>
        <circle cx="100" cy="100" r="5" fill="#1e293b"/>
        <text x="12"  y="114" font-size="9" fill="#94a3b8">0</text>
        <text x="96"  y="114" font-size="9" fill="#94a3b8" text-anchor="middle">50</text>
        <text x="185" y="114" font-size="9" fill="#94a3b8" text-anchor="end">100</text>
      </svg>
      <div class="gscore" id="gscore" style="color:#16a34a">—</div>
      <div class="glabel" id="gstatus">AWAITING DATA</div>
    </div>

    <!-- PM Panel -->
    <div class="card">
      <div class="card-t">🔧 Predictive Maintenance</div>
      <div id="pm-badge" class="rbadge" style="background:#f1f5f9;color:var(--t3)">NO DATA</div>
      <div class="mrow"><span class="mname">Maintenance Required</span><span class="mval" id="m-maint">—</span></div>
      <div class="mrow"><span class="mname">Failure Probability</span><span class="mval" id="m-prob">—</span></div>
      <div class="mrow"><span class="mname">Risk Level</span><span class="mval" id="m-risk">—</span></div>
      <div class="mrow"><span class="mname">Module − Ambient ΔT</span><span class="mval" id="m-dt">—</span></div>
      <div class="mrow"><span class="mname">Power Deviation</span><span class="mval" id="m-dev">—</span></div>
      <div class="mrow"><span class="mname">Temp Stress Level</span><span class="mval" id="m-ts">—</span></div>
    </div>

    <!-- Forecast Panel -->
    <div class="card">
      <div class="card-t">⚡ Power Forecasting</div>
      <div class="mrow"><span class="mname">Forecasted AC Power</span><span class="mval" id="f-fc" style="color:var(--blue)">—</span></div>
      <div class="mrow"><span class="mname">Actual AC Power</span><span class="mval" id="f-ac" style="color:var(--green)">—</span></div>
      <div class="mrow"><span class="mname">DC Power</span><span class="mval" id="f-dc">—</span></div>
      <div class="mrow"><span class="mname">DC/AC Ratio</span><span class="mval" id="f-dcratio">—</span></div>
      <div class="mrow"><span class="mname">Inverter Efficiency</span><span class="mval" id="f-eff" style="color:var(--green)">—</span></div>
      <div class="mrow"><span class="mname">Performance Ratio</span><span class="mval" id="f-pr">—</span></div>
      <div class="mrow"><span class="mname">Irradiation</span><span class="mval" id="f-irr">—</span></div>
    </div>
  </div>

  <!-- CHARTS -->
  <div class="charts">
    <div class="card">
      <div class="card-t">⚡ Live AC Power — Actual vs Forecast (Single Inverter)</div>
      <div class="cwrap"><canvas id="pc"></canvas></div>
    </div>
    <div class="card">
      <div class="card-t">🩺 Maintenance Probability %</div>
      <div class="cwrap"><canvas id="mc"></canvas></div>
    </div>
  </div>

  <!-- BOTTOM -->
  <div class="bot">
    <div class="card">
      <div class="card-t">📈 Health Index Trend</div>
      <div class="cwrap" style="height:165px"><canvas id="hc"></canvas></div>
    </div>
    <div class="card">
      <div class="card-t">🚨 Alert Feed</div>
      <div class="alist" id="alist">
        <div class="no-a">No alerts — select inverter &amp; start stream.</div>
      </div>
    </div>
  </div>
</div>

<footer class="footer">
  <div class="fstats">
    <div class="fstat"><span>Protocol</span><span>MQTT (In-Process) · solar/stream</span></div>
    <div class="fstat"><span>Models</span><span>RF Regressor + RF Classifier · Per-Inverter</span></div>
    <div class="fstat" id="fts"><span>Timestamp</span><span id="fts-val">—</span></div>
  </div>
  <div id="finv" style="color:var(--t3);font-weight:600;">Inverter: —</div>
</footer>

<script>
// ── Charts ─────────────────────────────────────────────────────────
const C0={responsive:true,maintainAspectRatio:false,animation:{duration:180},
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#475569',font:{size:10},boxWidth:10,padding:10}}},
  scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:8,font:{size:9}},grid:{color:'#f1f5f9'}},
          y:{ticks:{color:'#94a3b8',font:{size:9}},grid:{color:'#f1f5f9'}}}};
function mkChart(id,dsets){
  return new Chart(document.getElementById(id),{type:'line',
    data:{labels:[],datasets:dsets},options:JSON.parse(JSON.stringify(C0))});
}
const MAX=60;
const pc=mkChart('pc',[
  {label:'Actual AC (kW)', data:[],borderColor:'#16a34a',backgroundColor:'rgba(22,163,74,.07)',tension:.4,borderWidth:2,pointRadius:0,fill:true},
  {label:'Forecast (kW)', data:[],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,.05)',tension:.4,borderWidth:2,pointRadius:0,fill:true,borderDash:[5,3]}
]);
const mc=mkChart('mc',[{label:'Failure Prob %',data:[],borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.08)',tension:.4,borderWidth:2,pointRadius:0,fill:true}]);
const hc=mkChart('hc',[{label:'Health Index',data:[],borderColor:'#16a34a',backgroundColor:'rgba(22,163,74,.08)',tension:.4,borderWidth:2,pointRadius:0,fill:true}]);

function push(chart,lbl,vals){
  chart.data.labels.push(lbl);
  vals.forEach((v,i)=>chart.data.datasets[i].data.push(v));
  if(chart.data.labels.length>MAX){chart.data.labels.shift();chart.data.datasets.forEach(d=>d.data.shift());}
  chart.update('none');
}
function clearCharts(){[pc,mc,hc].forEach(c=>{c.data.labels=[];c.data.datasets.forEach(d=>d.data=[]);c.update('none');});}

// ── Gauge ──────────────────────────────────────────────────────────
function updateGauge(pct){
  pct=Math.max(0,Math.min(100,pct));
  const r=85,cx=100,cy=100;
  const endRad=Math.PI-(pct/100)*Math.PI;
  const ex=cx+r*Math.cos(endRad), ey=cy-r*Math.sin(endRad);
  const la=pct>50?1:0;
  document.getElementById('garc').setAttribute('d',
    `M15,100 A${r},${r},0,${la},1,${ex.toFixed(1)},${ey.toFixed(1)}`);
  const col=pct>=80?'#16a34a':pct>=60?'#d97706':pct>=40?'#ea580c':'#dc2626';
  document.getElementById('garc').setAttribute('stroke',col);
  document.getElementById('gneedle').setAttribute('transform',`rotate(${-90+pct*1.8},100,100)`);
  document.getElementById('gscore').textContent=pct.toFixed(1);
  document.getElementById('gscore').style.color=col;
}

// ── Toasts & alerts ────────────────────────────────────────────────
let notif=false;
if('Notification' in window){
  if(Notification.permission==='granted') notif=true;
  else if(Notification.permission==='default')
    Notification.requestPermission().then(p=>notif=p==='granted');
}
function showToast(d){
  const tc=document.getElementById('tc');
  const t=document.createElement('div'); t.className='toast';
  t.innerHTML=`<div class="toast-t">⚠ ${d.risk_level} RISK · Inverter ${(d.inverter||'').slice(-6)}</div>
  <div class="toast-b">Health: ${d.health_index} · Prob: ${d.maint_prob}% · ΔT: ${d.temp_delta}°C</div>`;
  tc.prepend(t); setTimeout(()=>t.remove(),6000);
  if(notif) new Notification(`⚠ ${d.risk_level} RISK – Inverter Alert`,
    {body:`Health ${d.health_index} | Fail prob ${d.maint_prob}%`});
}
let aCnt=0, lastAlertRow=0;
function addAlert(d){
  const list=document.getElementById('alist');
  list.querySelector('.no-a')?.remove();
  if(aCnt>=30) list.lastElementChild?.remove();
  const ts=new Date().toLocaleTimeString();
  const el=document.createElement('div'); el.className='aitem';
  el.innerHTML=`<span style="font-size:14px">⚠️</span>
  <div><div class="at">${d.risk_level} RISK · Inv ${(d.inverter||'').slice(-6)}</div>
  <div class="am">${ts} · Health: ${d.health_index}</div>
  <div class="ab">Prob: ${d.maint_prob}% | Dev: ${d.power_deviation>=0?'+':''}${d.power_deviation}% | Temp: ${d.temp_stress}</div></div>`;
  list.prepend(el); aCnt++;
}

// ── Color map ──────────────────────────────────────────────────────
const RC={LOW:'#16a34a',MODERATE:'#d97706',HIGH:'#ea580c',CRITICAL:'#dc2626'};

// ── UI update ──────────────────────────────────────────────────────
let rowCount=0, lastARow=0;
function upd(d){
  if(d.keepalive) return;
  rowCount++;
  document.getElementById('rctr').textContent=`Row ${rowCount}`;
  document.getElementById('fts-val').textContent=d.timestamp||'';
  document.getElementById('finv').textContent=`Inverter: ${(d.inverter||'').slice(-8)}`;
  document.getElementById('inv-badge-disp').textContent=`Inverter: ${(d.inverter||'').slice(-8)}`;

  const hi=d.health_index;
  const hc2=hi>=80?'#16a34a':hi>=60?'#d97706':hi>=40?'#ea580c':'#dc2626';
  document.getElementById('k-hi').textContent=hi;
  document.getElementById('k-hi').style.color=hc2;

  const mf=d.maint_required;
  const mEl=document.getElementById('k-mf');
  mEl.textContent=mf?'YES':'NO'; mEl.style.color=mf?'#dc2626':'#16a34a';

  const rc=RC[d.risk_level]||'#94a3b8';
  const rb=document.getElementById('k-rb');
  rb.textContent=d.risk_level;
  rb.style.cssText=`background:${rc}18;color:${rc};border:1px solid ${rc}44;`;

  document.getElementById('k-fc').textContent=d.forecast_kw.toFixed(0);
  document.getElementById('k-ac').textContent=d.actual_kw.toFixed(0);

  const pr=d.perf_ratio;
  const prEl=document.getElementById('k-pr');
  prEl.textContent=pr.toFixed(1)+'%';
  prEl.style.color=pr>=90?'#16a34a':pr>=70?'#d97706':'#dc2626';

  const eff=d.efficiency_pct;
  const effEl=document.getElementById('k-eff');
  effEl.textContent=eff.toFixed(1)+'%';
  effEl.style.color=eff>=90?'#16a34a':eff>=80?'#d97706':'#dc2626';

  document.getElementById('k-loss').textContent=d.power_loss_kw.toFixed(0);
  document.getElementById('k-irr').textContent=(d.irradiation*1000).toFixed(0);

  updateGauge(hi);
  const gsEl=document.getElementById('gstatus');
  gsEl.textContent=d.health_status; gsEl.style.color=hc2;

  const pmb=document.getElementById('pm-badge');
  pmb.textContent=mf?`⚠ ${d.risk_level} — MAINTENANCE REQUIRED`:`✔ ${d.risk_level} — SYSTEM NOMINAL`;
  pmb.style.cssText=`background:${(mf?rc:'#16a34a')}15;color:${mf?rc:'#16a34a'};border:1.5px solid ${(mf?rc:'#16a34a')}33;`;

  document.getElementById('m-maint').textContent=mf?'⚠ YES':'✔ NO';
  document.getElementById('m-maint').style.color=mf?'#dc2626':'#16a34a';
  document.getElementById('m-prob').textContent=d.maint_prob+'%';
  document.getElementById('m-prob').style.color=rc;
  document.getElementById('m-risk').textContent=d.risk_level;
  document.getElementById('m-risk').style.color=rc;
  document.getElementById('m-dt').textContent=d.temp_delta.toFixed(2)+' °C';
  const devEl=document.getElementById('m-dev');
  devEl.textContent=(d.power_deviation>=0?'+':'')+d.power_deviation.toFixed(1)+'%';
  devEl.style.color=d.power_deviation>=0?'#16a34a':'#dc2626';
  const tsEl=document.getElementById('m-ts');
  tsEl.textContent=d.temp_stress;
  tsEl.style.color={NORMAL:'#16a34a',MODERATE:'#d97706',HIGH:'#ea580c',CRITICAL:'#dc2626'}[d.temp_stress]||'#475569';

  document.getElementById('f-fc').textContent=d.forecast_kw.toFixed(2)+' kW';
  document.getElementById('f-ac').textContent=d.actual_kw.toFixed(2)+' kW';
  document.getElementById('f-dc').textContent=d.dc_power.toFixed(2)+' kW';
  document.getElementById('f-dcratio').textContent=d.dc_ac_ratio.toFixed(3);
  document.getElementById('f-eff').textContent=d.efficiency_pct.toFixed(1)+'%';
  document.getElementById('f-pr').textContent=d.perf_ratio.toFixed(1)+'%';
  document.getElementById('f-irr').textContent=(d.irradiation*1000).toFixed(0)+' W/m²';

  const ts=(d.timestamp||'').slice(11,16);
  push(pc,ts,[d.actual_kw,d.forecast_kw]);
  push(mc,ts,[d.maint_prob]);
  push(hc,ts,[d.health_index]);

  if(mf && rowCount-lastARow>=5){ addAlert(d); showToast(d); lastARow=rowCount; }
}

// ── Inverter selector ──────────────────────────────────────────────
function loadInverters(){
  fetch('/inverters').then(r=>r.json()).then(data=>{
    const sel=document.getElementById('inv-select');
    sel.innerHTML='';
    data.inverters.forEach(k=>{
      const o=document.createElement('option');
      o.value=k; o.textContent=`Inv ${k.slice(-8)}`;
      if(k===data.active) o.selected=true;
      sel.appendChild(o);
    });
  });
}
function selectInverter(key){
  if(!key) return;
  stopStream();
  clearCharts(); rowCount=0; aCnt=0; lastARow=0;
  document.getElementById('rctr').textContent='Row 0';
  document.getElementById('alist').innerHTML='<div class="no-a">Switched inverter — start streaming.</div>';
  fetch(`/select/${key}`).then(()=>{ document.getElementById('finv').textContent=`Inverter: ${key.slice(-8)}`; });
}

// ── Stream control ─────────────────────────────────────────────────
let es=null;
function setStatus(on){
  document.getElementById('mdot').className='dot'+(on?' on':'');
  document.getElementById('mstatus').textContent=on?'Streaming':'Idle';
  document.getElementById('btn-s').style.display=on?'none':'flex';
  document.getElementById('btn-x').style.display=on?'flex':'none';
}
function startStream(){
  fetch('/start').then(()=>{
    if(es) es.close();
    es=new EventSource('/stream');
    setStatus(true);
    es.onmessage=e=>{try{upd(JSON.parse(e.data));}catch{}};
    es.onerror=()=>setStatus(false);
  });
}
function stopStream(){
  fetch('/stop').then(()=>{ if(es){es.close();es=null;} setStatus(false); });
}

// ── Speed ──────────────────────────────────────────────────────────
function setSpeed(v){
  const s=((21-v)*0.1).toFixed(1);
  document.getElementById('spd-val').textContent=s+'s';
  fetch(`/speed/${s}`);
}
setSpeed(8);
loadInverters();
</script>
</body>
</html>"""

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
