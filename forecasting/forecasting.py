import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# LOAD DATA
# -------------------------------
gen = pd.read_csv(os.path.join(current_dir, "Plant_1_Generation_Data (1).csv"))
weather = pd.read_csv(os.path.join(current_dir, "Plant_1_Weather_Sensor_Data.csv"))

gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], dayfirst=True, format='mixed')
weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], format='mixed')

gen = gen.groupby('DATE_TIME')[['AC_POWER']].sum().reset_index()

df = pd.merge(gen, weather, on='DATE_TIME')

# Remove night values
df = df[df['IRRADIATION'] > 0].reset_index(drop=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Lag features (IMPROVED)
for lag in range(1, 6):
    df[f'POWER_LAG_{lag}'] = df['AC_POWER'].shift(lag)

# Time features
df['HOUR'] = df['DATE_TIME'].dt.hour

# Cyclic encoding (VERY IMPORTANT)
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)

df = df.dropna()

# -------------------------------
# FEATURES & TARGET
# -------------------------------
FEATURES = [
    'IRRADIATION',
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'POWER_LAG_1',
    'POWER_LAG_2',
    'POWER_LAG_3',
    'POWER_LAG_4',
    'POWER_LAG_5',
    'HOUR_SIN',
    'HOUR_COS'
]

X = df[FEATURES]
y = df['AC_POWER']

# -------------------------------
# SPLIT (TIME-SERIES SAFE)
# -------------------------------
split1 = int(0.7 * len(df))
split2 = int(0.9 * len(df))

X_train, y_train = X[:split1], y[:split1]
X_val, y_val = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]

# -------------------------------
# MODEL
# -------------------------------
model = RandomForestRegressor(
    n_estimators=120,
    max_depth=12,
    min_samples_split=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate(X, y, name):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    print(f"\n{name} Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    return pred

train_pred = evaluate(X_train, y_train, "Train")
val_pred = evaluate(X_val, y_val, "Validation")
test_pred = evaluate(X_test, y_test, "Test")

# -------------------------------
# PLOT (FORECAST)
# -------------------------------
plt.style.use('default')

plt.figure()
plt.plot(y_test.values, label="Actual Power", linewidth=2)
plt.plot(test_pred, label="Predicted Power", linestyle='--')

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Random Forest Power Forecasting (Final)")
plt.legend()

plt.savefig(os.path.join(current_dir, "rf_forecast.png"))
plt.close()

# -------------------------------
# ERROR PLOT (IMPORTANT)
# -------------------------------
plt.figure()
plt.plot(y_test.values - test_pred)
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Prediction Error (Final)")

plt.savefig(os.path.join(current_dir, "rf_error.png"))
plt.close()

print("\nForecasting completed! Graphs saved.")
