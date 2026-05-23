import sys
import subprocess
import os

# ─────────────────────────────────────────────
# FORCE RUN WITH PYTHON 3.10
# ─────────────────────────────────────────────
if "Python310" not in sys.executable:
    print("🔁 Restarting with Python 3.10...")

    python310_path = r"C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe"

    if os.path.exists(python310_path):
        try:
            subprocess.run([python310_path, __file__])
            sys.exit(0)
        except Exception as e:
            print(f"❌ Failed to relaunch: {e}")
            print("⚠️ Continuing with current Python (may fail)")
    else:
        print("❌ Python 3.10 not found at specified path")
        print("⚠️ Please install Python 3.10 or update the path")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 1. LOAD / SIMULATE DATA
# ─────────────────────────────────────────────
DATA_PATH = "pv_anomaly_data.csv"

def load_or_simulate(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        np.random.seed(42)
        class_counts = {0: 20000, 1: 2000, 2: 3000, 3: 2000, 4: 4000}
        stats = {
            0:(2.16,3,1.9,2.9,247,340,22.5,14,133,136,133,136),
            1:(8.5,0.8,8.2,0.8,300,150,28,5,0.5,0.3,0.5,0.3),
            2:(0.05,0.02,0.01,0.005,280,130,26,6,360,5,360,5),
            3:(1.5,2.5,1.3,2.4,240,330,22,14,100,120,100,120),
            4:(1,2,0.8,1.8,150,200,20,12,90,110,90,110)
        }

        rows=[]
        for cls,count in class_counts.items():
            s=stats[cls]
            for _ in range(count):
                rows.append([
                    cls,
                    np.random.normal(s[0],s[1]),
                    np.random.normal(s[2],s[3]),
                    np.random.normal(s[4],s[5]),
                    np.random.normal(s[6],s[7]),
                    np.random.normal(s[8],s[9]),
                    np.random.normal(s[10],s[11])
                ])

        df=pd.DataFrame(rows,columns=['result','idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2'])
        return df.sample(frac=1).reset_index(drop=True)

df = load_or_simulate(DATA_PATH)

# ─────────────────────────────────────────────
# 2. RAW FEATURES ONLY
# ─────────────────────────────────────────────
df = df[['result','idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2']]

X_raw = df.drop('result', axis=1).values
y_raw = df['result'].values

# Add noise
X_raw = X_raw + np.random.normal(0, 0.05, X_raw.shape)

# ─────────────────────────────────────────────
# 3. CREATE SEQUENCES
# ─────────────────────────────────────────────
SEQ_LEN = 60
STEP = 30

def create_sequences(X, y):
    Xs, ys = [], []
    for i in range(0, len(X) - SEQ_LEN, STEP):
        Xs.append(X[i:i+SEQ_LEN])
        ys.append(np.bincount(y[i:i+SEQ_LEN]).argmax())
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_raw, y_raw)

# ─────────────────────────────────────────────
# 4. BALANCE SEQUENCES
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# IMPROVED BALANCING (NOT TOO AGGRESSIVE)
# ─────────────────────────────────────────────
seq_df = pd.DataFrame({'label': y_seq})
seq_df['idx'] = np.arange(len(y_seq))

# Set minimum threshold (avoid too small dataset)
TARGET_SAMPLES = 500  # adjust if needed

balanced_idx = []

for cls in seq_df['label'].unique():
    cls_samples = seq_df[seq_df['label'] == cls]

    if len(cls_samples) > TARGET_SAMPLES:
        cls_bal = resample(cls_samples,
                          replace=False,
                          n_samples=TARGET_SAMPLES,
                          random_state=42)
    else:
        cls_bal = cls_samples  # keep all if already small

    balanced_idx.extend(cls_bal['idx'].values)

balanced_idx = np.array(balanced_idx)

X_seq = X_seq[balanced_idx]
y_seq = y_seq[balanced_idx]

# ─────────────────────────────────────────────
# 5. SCALE + SPLIT
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
X_seq = X_flat.reshape(X_seq.shape)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_seq, y_seq, test_size=0.1, stratify=y_seq, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42)

y_train_cat = to_categorical(y_train, 5)
y_val_cat = to_categorical(y_val, 5)

# ─────────────────────────────────────────────
# 6. LSTM MODEL
# ─────────────────────────────────────────────
inp = Input(shape=(SEQ_LEN, X_seq.shape[2]))

x = LSTM(64, return_sequences=True)(inp)
x = Dropout(0.3)(x)

x = LSTM(32)(x)
x = Dense(32, activation='relu')(x)

out = Dense(5, activation='softmax')(x)

model = Model(inp, out)

model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    batch_size=128,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# ─────────────────────────────────────────────
# 8. PREDICTIONS
# ─────────────────────────────────────────────
y_pred_train = np.argmax(model.predict(X_train), axis=1)
y_pred_val = np.argmax(model.predict(X_val), axis=1)
y_pred_test = np.argmax(model.predict(X_test), axis=1)

print("\nTest Classification Report:\n")
print(classification_report(
    y_test,
    y_pred_test,
    target_names=['Normal','Short','Open','Degrad','Shading']
))

# ─────────────────────────────────────────────
# 9. LIGHT COLOR CONFUSION MATRICES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1,3, figsize=(18,5))

ConfusionMatrixDisplay.from_predictions(
    y_train, y_pred_train, ax=axes[0],
    cmap='PuBu', colorbar=False
)
axes[0].set_title("Train CM")

ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred_val, ax=axes[1],
    cmap='PuBu', colorbar=False
)
axes[1].set_title("Validation CM")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_test, ax=axes[2],
    cmap='PuBu', colorbar=True
)
axes[2].set_title("Test CM")

for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig("lstm_final_cm.png", dpi=300)

# ─────────────────────────────────────────────
# 10. PR CURVE
# ─────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4])
y_prob = model.predict(X_test)

plt.figure(figsize=(6,5))
for i in range(5):
    p, r, _ = precision_recall_curve(y_test_bin[:,i], y_prob[:,i])
    plt.plot(r, p, label=str(i))

plt.legend(title="Classes")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.grid(True)
plt.savefig("lstm_final_pr.png", dpi=300)

# ─────────────────────────────────────────────
# 11. SAVE MODEL
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/lstm_final_model.h5")

print("\n✅ Done! Model saved + clean light CM generated")
