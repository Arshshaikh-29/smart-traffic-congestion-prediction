import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Synthetic demo data only. Replace with measured traffic data for final evaluation.
rng = np.random.default_rng(42)
n = 10000

hour = rng.integers(0, 24, n)
day = rng.integers(0, 7, n)
temperature = rng.uniform(15, 42, n)
rain = rng.uniform(0, 30, n)
vehicles = rng.integers(100, 2500, n)
road_capacity = rng.integers(500, 2000, n)
accident = rng.integers(0, 2, n)

rush_hour = (((hour >= 8) & (hour <= 10)) |
             ((hour >= 17) & (hour <= 20))).astype(int)

speed = (65 - vehicles / road_capacity * 20 - rush_hour * 15
         - rain * 0.3 - accident * 20 + rng.normal(0, 4, n))
speed = np.clip(speed, 5, 90)

X = pd.DataFrame({
    "hour": hour,
    "day": day,
    "temperature": temperature,
    "rain": rain,
    "vehicles": vehicles,
    "road_capacity": road_capacity,
    "accident": accident
})
y = speed

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, pred):.2f} km/h")
print(f"R2: {r2_score(y_test, pred):.3f}")
joblib.dump(model, "traffic_model.pkl")
print("Saved traffic_model.pkl")
