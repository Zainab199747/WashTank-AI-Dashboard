import pandas as pd

from sklearn.neural_network import MLPRegressor
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

df_mlp = df[[target] + features].dropna()

print("\nRows Used:", len(df_mlp))

X = df_mlp[features]

y = df_mlp[target]

# ==========================
# MLP MODEL
# ==========================

model = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=5000,
    random_state=42
)

model.fit(X, y)

# ==========================
# PREDICTIONS
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

print("\nMLP RESULTS")

print("R² =", round(r2,4))

print("RMSE =", round(rmse,4))

# ==========================
# RESULTS TABLE
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

plt.title("MLP Neural Network: Actual vs Predicted")

plt.show()