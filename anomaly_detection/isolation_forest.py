import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD / SIMULATE DATA
# ─────────────────────────────────────────────
DATA_PATH = "pv_anomaly_data.csv"

def load_or_simulate(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        np.random.seed(42)
        class_counts = {0: 886884, 1: 1226, 2: 5695, 3: 1204, 4: 153566}
        rows = []
        stats = {
            0: (2.16,3.0,1.90,2.95,247,340,22.5,14,133,136,133,136),
            1: (8.5,0.8,8.2,0.8,300,150,28,5,0.5,0.3,0.5,0.3),
            2: (0.05,0.02,0.01,0.005,280,130,26,6,360,5,360,5),
            3: (1.5,2.5,1.3,2.4,240,330,22,14,100,120,100,120),
            4: (1.0,2.0,0.8,1.8,150,200,20,12,90,110,90,110),
        }
        for cls,count in class_counts.items():
            s=stats[cls]
            idc1=np.random.normal(s[0],s[1],count)
            idc2=np.random.normal(s[2],s[3],count)
            irra=np.random.normal(s[4],s[5],count)
            pvtemp=np.random.normal(s[6],s[7],count)
            vdc1=np.random.normal(s[8],s[9],count)
            vdc2=np.random.normal(s[10],s[11],count)
            for i in range(count):
                rows.append([cls,idc1[i],idc2[i],irra[i],pvtemp[i],vdc1[i],vdc2[i]])
        df=pd.DataFrame(rows,columns=['result','idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2'])
        df=df.sample(frac=1).reset_index(drop=True)
    return df

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

X=df[FEATURES].values
y=df['result'].values

# ─────────────────────────────────────────────
# 3. CLASS WEIGHTS
# ─────────────────────────────────────────────
classes=np.array([0,1,2,3,4])
cw=compute_class_weight('balanced',classes=classes,y=y)
class_weight_dict=dict(zip(classes,cw))
sample_weights=np.array([class_weight_dict[i] for i in y])

# ─────────────────────────────────────────────
# 4. SPLIT (70 / 20 / 10)
# ─────────────────────────────────────────────
X_temp,X_test,y_temp,y_test,sw_temp,sw_test=train_test_split(
    X,y,sample_weights,test_size=0.1,stratify=y,random_state=42)

X_train,X_val,y_train,y_val,sw_train,sw_val=train_test_split(
    X_temp,y_temp,sw_temp,test_size=0.2222,stratify=y_temp,random_state=42)

scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_val_sc=scaler.transform(X_val)
X_test_sc=scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────
anomaly_fraction=(y!=0).sum()/len(y)

model=IsolationForest(n_estimators=200,contamination=anomaly_fraction,random_state=42)
model.fit(X_train_sc)

# ─────────────────────────────────────────────
# 6. SCORES
# ─────────────────────────────────────────────
train_scores=-model.decision_function(X_train_sc)
val_scores=-model.decision_function(X_val_sc)
test_scores=-model.decision_function(X_test_sc)

y_train_bin=(y_train!=0).astype(int)
y_val_bin=(y_val!=0).astype(int)
y_test_bin=(y_test!=0).astype(int)

# ─────────────────────────────────────────────
# 7. THRESHOLD
# ─────────────────────────────────────────────
precisions,recalls,thresholds=precision_recall_curve(y_test_bin,test_scores)

f1=2*(precisions*recalls)/(precisions+recalls+1e-8)
best_idx=np.argmax(f1)
best_threshold=thresholds[best_idx]

# predictions
y_pred_train=(train_scores>=best_threshold).astype(int)
y_pred_val=(val_scores>=best_threshold).astype(int)
y_pred_test=(test_scores>=best_threshold).astype(int)

# ─────────────────────────────────────────────
# 8. REPORT
# ─────────────────────────────────────────────
print(classification_report(y_test,y_pred_test))

# ─────────────────────────────────────────────
# 9. CONFUSION MATRICES
# ─────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(18,5))

ConfusionMatrixDisplay.from_predictions(y_train_bin,y_pred_train,ax=axes[0],cmap='Blues')
axes[0].set_title("Train CM")

ConfusionMatrixDisplay.from_predictions(y_val_bin,y_pred_val,ax=axes[1],cmap='Blues')
axes[1].set_title("Validation CM")

ConfusionMatrixDisplay.from_predictions(y_test_bin,y_pred_test,ax=axes[2],cmap='Blues')
axes[2].set_title("Test CM")

plt.tight_layout()
plt.show()
plt.savefig("models/step1_cm.png", bbox_inches='tight')

# ─────────────────────────────────────────────
# 10. PR CURVE
# ─────────────────────────────────────────────
ap=average_precision_score(y_test_bin,test_scores)

plt.figure(figsize=(6,5))
plt.plot(recalls,precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR Curve (AP={ap:.3f})")
plt.grid()
plt.show()
plt.savefig("models/step1_pr_curve.png", bbox_inches='tight')

# ─────────────────────────────────────────────
# 11. REQUIRED GRAPHS
# ─────────────────────────────────────────────

# Fault class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['result'])
plt.title("Fault Class Distribution")
plt.show()
plt.savefig("models/step1_class_dist.png", bbox_inches='tight')

# Feature distributions
df[['idc_1','idc_2','irra','pvtemp','vdc_1','vdc_2']].hist(figsize=(12,8),bins=40)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()
plt.savefig("models/step1_feat_dist.png", bbox_inches='tight')

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[FEATURES].corr(),cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
plt.savefig("models/step1_heatmap.png", bbox_inches='tight')
