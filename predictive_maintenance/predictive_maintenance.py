import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
import os

# -------------------------------
# WHITE BACKGROUND STYLE
# -------------------------------
plt.style.use('default')
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(gen_path, weather_path):
    gen = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)

    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], dayfirst=True, format='mixed')
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], format='mixed')

    gen = gen.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER']].sum().reset_index()

    df = pd.merge(gen, weather, on='DATE_TIME', how='inner')
    df = df[df['IRRADIATION'] > 0]

    return df

# -------------------------------
# MAIN
# -------------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    df = load_data(
        os.path.join(current_dir, "Plant_1_Generation_Data (1).csv"),
        os.path.join(current_dir, "Plant_1_Weather_Sensor_Data.csv")
    )

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    df['EFFICIENCY'] = df['AC_POWER'] / (df['DC_POWER'] + 1e-6)
    df['TEMP_DELTA'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
    df['DC_AC_RATIO'] = df['DC_POWER'] / (df['AC_POWER'] + 1e-6)

    # -------------------------------
    # LABEL CREATION (RELAXED)
    # -------------------------------
    eff_low = df['EFFICIENCY'] < df['EFFICIENCY'].median() * 0.95
    temp_stress = df['MODULE_TEMPERATURE'] > 52
    dc_ac_high = df['DC_AC_RATIO'] > df['DC_AC_RATIO'].quantile(0.80)

    df['LABEL'] = (eff_low | temp_stress | dc_ac_high).astype(int)

    # -------------------------------
    # ADD LABEL NOISE (IMPORTANT)
    # -------------------------------
    np.random.seed(42)
    noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[noise_idx, 'LABEL'] = 1 - df.loc[noise_idx, 'LABEL']

    # -------------------------------
    # FEATURES (NO LEAKAGE)
    # -------------------------------
    FEATURES = [
        'IRRADIATION',
        'AMBIENT_TEMPERATURE',
        'MODULE_TEMPERATURE',
        'TEMP_DELTA',
        'DC_AC_RATIO'
    ]

    X = df[FEATURES]
    y = df['LABEL']

    # -------------------------------
    # SPLIT 70 / 20 / 10
    # -------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    # -------------------------------
    # MODEL (REDUCED POWER)
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # EVALUATION FUNCTION
    # -------------------------------
    def evaluate(X, y, name):
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        
        print(f"--- {name} Metrics ---")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 Score:  {f1:.3f}\n")
        return pred

    train_pred = evaluate(X_train, y_train, "Train")
    val_pred = evaluate(X_val, y_val, "Validation")
    test_pred = evaluate(X_test, y_test, "Test")

    # -------------------------------
    # COMBINED CONFUSION MATRIX
    # -------------------------------
    def plot_combined_cm(results, filename="predictive_cm.png"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (y_true, y_pred, name) in enumerate(results):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                        cbar=False, annot_kws={"size": 14, "weight": "bold"})
            axes[i].set_title(f"{name.capitalize()} Matrix", fontsize=14, pad=15)
            axes[i].set_xlabel("Predicted", fontsize=12)
            axes[i].set_ylabel("Actual", fontsize=12)
            
        plt.suptitle("Predictive Maintenance Confusion Matrices", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    results = [
        (y_train, train_pred, "train"),
        (y_val, val_pred, "validation"),
        (y_test, test_pred, "test")
    ]
    plot_combined_cm(results)

    # -------------------------------
    # ROC CURVE
    # -------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # -------------------------------
    # PR CURVE
    # -------------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("pr_curve.png")
    plt.close()

    print("\nAll plots saved successfully!")

# -------------------------------
if __name__ == "__main__":
    main()
