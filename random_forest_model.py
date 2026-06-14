import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# READ DATA
# ==========================

df = pd.read_csv("master_dataset.csv")

# ==========================
# FEATURES
# ==========================

features = [
    "avg_temperature",
    "avg_pressure",
    "avg_level",
    "avg_oil_outlet",
    "avg_water_outlet",
    "daily_dosage",
    "clean_water_points",
    "emulsion_points",
    "oil_points",
    "avg_thermal_range",
    "avg_hotspot_ratio",
    "avg_vertical_gradient",
    "avg_thermal_uniformity"
]

target = "emulsion_level_m"

# ==========================
# DATASET
# ==========================

df_rf = df[[target] + features].dropna()

print("\nRows Used:", len(df_rf))

print("\nDataset Used:")
print(df_rf)

# ==========================
# X AND Y
# ==========================

X = df_rf[features]

y = df_rf[target]

# ==========================
# RANDOM FOREST MODEL
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X)

# ==========================
# EVALUATION
# ==========================

r2 = r2_score(y, y_pred)

rmse = mean_squared_error(
    y,
    y_pred
) ** 0.5

print("\nRANDOM FOREST RESULTS")

print("R² =", round(r2, 4))

print("RMSE =", round(rmse, 4))

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Important Variables:")

print(importance)

# ==========================
# ACTUAL VS PREDICTED
# ==========================

results = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred
})

print("\nPredictions:")

print(results)

# ==========================
# PLOT
# ==========================

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

plt.scatter(y, y_pred)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--'
)

plt.xlabel("Actual Emulsion Level")

plt.ylabel("Predicted Emulsion Level")

plt.title("Random Forest: Actual vs Predicted")

plt.show()

# ==========================
# FEATURE IMPORTANCE PLOT
# ==========================

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.show()