import sys
import subprocess
import os

# Auto-relaunch script using the correct Python 3.10 environment
if "Python310" not in sys.executable:
    print("Restarting with Python 3.10 Environment...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    subprocess.run([r"C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe", __file__])
    sys.exit(0)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, Concatenate, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# CREATE FOLDER
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "pv_anomaly_data.csv"

def load_or_simulate(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        np.random.seed(42)
        class_counts = {0: 886884, 1: 1226, 2: 5695, 3: 1204, 4: 153566}
        rows=[]
        for cls,count in class_counts.items():
            for _ in range(count):
                rows.append([
                    cls,
                    np.random.uniform(0,10),
                    np.random.uniform(0,10),
                    np.random.uniform(0,1000),
                    np.random.uniform(0,60),
                    np.random.uniform(0,400),
                    np.random.uniform(0,400)
                ])
        df=pd.DataFrame(rows,columns=['result','idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2'])
        return df.sample(frac=1).reset_index(drop=True)

df = load_or_simulate(DATA_PATH)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['power_1']=df['idc_1']*df['vdc_1']
df['power_2']=df['idc_2']*df['vdc_2']
df['total_power']=df['power_1']+df['power_2']
df['current_ratio']=df['idc_1']/(df['idc_2']+1e-6)
df['voltage_ratio']=df['vdc_1']/(df['vdc_2']+1e-6)
df['pr_ratio']=df['total_power']/(df['irra']+1e-6)
df['idc_imbalance']=abs(df['idc_1']-df['idc_2'])
df['vdc_imbalance']=abs(df['vdc_1']-df['vdc_2'])

FEATURES=[c for c in df.columns if c!='result']
X_raw=df[FEATURES].values
y_raw=df['result'].values

# ─────────────────────────────────────────────
# 3. CLASS WEIGHTS
# ─────────────────────────────────────────────
cw=compute_class_weight('balanced',classes=np.arange(5),y=y_raw)
cw_dict=dict(zip(range(5),cw))

class_names={0:'Normal',1:'Short',2:'Open',3:'Degrad',4:'Shading'}

# ─────────────────────────────────────────────
# 4. ISOLATION FOREST
# ─────────────────────────────────────────────
scaler_if=StandardScaler()
X_sc=scaler_if.fit_transform(X_raw)

IF=IsolationForest(contamination=(y_raw!=0).sum()/len(y_raw))
IF.fit(X_sc)

scores=-IF.decision_function(X_sc)
scores=(scores-scores.min())/(scores.max()-scores.min())

X_enriched=np.concatenate([X_sc,scores.reshape(-1,1)],axis=1)

# ─────────────────────────────────────────────
# SAVE IF + SCALER
# ─────────────────────────────────────────────
joblib.dump(IF, "models/isolation_forest.pkl")
joblib.dump(scaler_if, "models/scaler.pkl")

# ─────────────────────────────────────────────
# 5. SEQUENCES
# ─────────────────────────────────────────────
SEQ=30
STEP=10

def create_seq(X,y):
    Xs,ys=[],[]
    for i in range(0,len(X)-SEQ,STEP):
        Xs.append(X[i:i+SEQ])
        ys.append(y[i+SEQ-1])
    return np.array(Xs),np.array(ys)

X_seq,y_seq=create_seq(X_enriched,y_raw)

# ─────────────────────────────────────────────
# 6. SPLIT (70/20/10)
# ─────────────────────────────────────────────
X_temp,X_test,y_temp,y_test=train_test_split(X_seq,y_seq,test_size=0.1,stratify=y_seq)
X_train,X_val,y_train,y_val=train_test_split(X_temp,y_temp,test_size=0.2222,stratify=y_temp)

y_train_cat=to_categorical(y_train,5)
y_val_cat=to_categorical(y_val,5)
y_test_cat=to_categorical(y_test,5)

# ─────────────────────────────────────────────
# 7. MODEL
# ─────────────────────────────────────────────
inp=Input(shape=(SEQ,X_seq.shape[2]))

x=Bidirectional(LSTM(128,return_sequences=True))(inp)
x=BatchNormalization()(x)
x=Bidirectional(LSTM(64))(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.3)(x)

if_score=Lambda(lambda t: t[:,-1,-1:])(inp)
if_branch=Dense(16,activation='relu')(if_score)

x=Concatenate()([x,if_branch])
x=Dense(48,activation='relu')(x)
out=Dense(5,activation='softmax')(x)

model=Model(inp,out)

model.compile(optimizer=Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
model.fit(X_train,y_train_cat,
          validation_data=(X_val,y_val_cat),
          epochs=20,
          batch_size=256,
          class_weight=cw_dict,
          callbacks=[EarlyStopping(patience=5,restore_best_weights=True)])

# ─────────────────────────────────────────────
# SAVE LSTM MODEL
# ─────────────────────────────────────────────
model.save("models/hybrid_lstm_model.keras")

# ─────────────────────────────────────────────
# 9. PREDICTIONS
# ─────────────────────────────────────────────
y_prob=model.predict(X_test)
y_pred_test=np.argmax(y_prob,axis=1)
y_pred_train=np.argmax(model.predict(X_train),axis=1)
y_pred_val=np.argmax(model.predict(X_val),axis=1)

print(classification_report(y_test,y_pred_test))

# ─────────────────────────────────────────────
# 10. CONFUSION MATRICES
# ─────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(18,5))

ConfusionMatrixDisplay.from_predictions(y_train,y_pred_train,ax=axes[0],cmap='Blues')
axes[0].set_title("Train CM")

ConfusionMatrixDisplay.from_predictions(y_val,y_pred_val,ax=axes[1],cmap='Blues')
axes[1].set_title("Validation CM")

ConfusionMatrixDisplay.from_predictions(y_test,y_pred_test,ax=axes[2],cmap='Blues')
axes[2].set_title("Test CM")

plt.tight_layout()
plt.savefig("models/confusion_matrices.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 11. PR CURVE
# ─────────────────────────────────────────────
y_test_bin=label_binarize(y_test,classes=[0,1,2,3,4])

plt.figure(figsize=(6,5))
for i in range(5):
    p,r,_=precision_recall_curve(y_test_bin[:,i],y_prob[:,i])
    plt.plot(r,p,label=class_names[i])

plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve (Hybrid)")
plt.grid()
plt.savefig("models/pr_curve.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 12. REQUIRED GRAPHS
# ─────────────────────────────────────────────

# Class distribution
plt.figure()
sns.countplot(x=df['result'])
plt.title("Fault Class Distribution")
plt.savefig("models/class_distribution.png", dpi=150)
plt.show()

# Feature distribution
df[['idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2']].hist(figsize=(12,8))
plt.tight_layout()
plt.savefig("models/feature_distribution.png", dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[FEATURES].corr(),cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("models/correlation_heatmap.png", dpi=150)
plt.show()

print("\n[✓] ALL MODELS & PLOTS SAVED IN 'models/' FOLDER")
