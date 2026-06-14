import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# قراءة البيانات
df = pd.read_csv("master_dataset.csv")

# اختيار الأيام التي تحتوي على كثافة
df2 = df[["emulsion_level_m",
          "avg_temperature",
          "avg_density"]].dropna()

# المتغيرات المستقلة
X = df2[["avg_temperature",
         "avg_density"]]

# الهدف
y = df2["emulsion_level_m"]

# إنشاء النموذج
model = LinearRegression()

# التدريب
model.fit(X, y)

# التنبؤ
y_pred = model.predict(X)

# المعاملات
a = model.intercept_
b1 = model.coef_[0]
b2 = model.coef_[1]

print("\nEmpirical Model:")

print(
f"E = {a:.4f} + "
f"({b1:.4f})*T + "
f"({b2:.4f})*D"
)

# تقييم النموذج
r2 = r2_score(y, y_pred)

print("\nR² =", round(r2,4))

print("\nNumber of rows used:")
print(len(df2))

print("\nDataset used:")
print(df2)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(y, y_pred)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--'
)

plt.xlabel("Actual Emulsion Level")
plt.ylabel("Predicted Emulsion Level")
plt.title("Actual vs Predicted")

plt.show()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df2["avg_temperature"],
    df2["avg_density"],
    df2["emulsion_level_m"]
)

T_range = np.linspace(
    df2["avg_temperature"].min(),
    df2["avg_temperature"].max(),
    20
)

D_range = np.linspace(
    df2["avg_density"].min(),
    df2["avg_density"].max(),
    20
)

T_grid, D_grid = np.meshgrid(T_range, D_range)

E_grid = (
    a
    + b1*T_grid
    + b2*D_grid
)

ax.plot_surface(
    T_grid,
    D_grid,
    E_grid,
    alpha=0.5
)

ax.set_xlabel("Temperature")
ax.set_ylabel("Density")
ax.set_zlabel("Emulsion Level")

plt.title("3D Empirical Model")
plt.show()
# =========================
# LEVEL 3 DATASET
# =========================

cols = [
    "emulsion_level_m",
    "avg_temperature",
    "avg_density",
    "avg_viscosity",
    "avg_chloride"
]

df3 = df[cols].dropna()

print("\nLevel 3 Dataset:")
print(df3)

print("\nNumber of rows used:")
print(len(df3))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ======================
# LEVEL 3
# ======================

X = df3[
    [
        "avg_temperature",
        "avg_density",
        "avg_viscosity"
    ]
]

y = df3["emulsion_level_m"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

a = model.intercept_

b1 = model.coef_[0]
b2 = model.coef_[1]
b3 = model.coef_[2]

print("\nLEVEL 3 MODEL")

print(
    f"E = {a:.4f} + "
    f"({b1:.4f})*T + "
    f"({b2:.4f})*D + "
    f"({b3:.4f})*V"
)

print(f"\nR² = {r2:.4f}")

import matplotlib.pyplot as plt

models = ["Level 1\n(T)", "Level 2\n(T+D)", "Level 3\n(T+D+V)"]
scores = [0.167, 0.947, 1.000]

plt.figure(figsize=(7,5))

plt.bar(models, scores)

plt.ylabel("R²")
plt.ylim(0, 1.1)

plt.title("Empirical Models Comparison")

for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

plt.show()

from sklearn.linear_model import LinearRegression

# حذف الأعمدة غير الرقمية
df_ml = df.select_dtypes(include=["number"])

# حذف الأعمدة التي تحتوي على قيم مفقودة
df_ml = df_ml.dropna(axis=1)

# الهدف
y = df_ml["emulsion_level_m"]

# المتغيرات
X = df_ml.drop(columns=["emulsion_level_m"])

# تدريب النموذج
model = LinearRegression()
model.fit(X, y)

# استخراج الأهمية
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["AbsCoeff"] = abs(importance["Coefficient"])

importance = importance.sort_values(
    "AbsCoeff",
    ascending=False
)

print("\nTop Important Variables:")
print(importance.head(15))






# ==========================
# LEVEL 4 MODEL
# ==========================

cols = [
    "emulsion_level_m",
    "avg_temperature",
    "avg_density",
    "avg_viscosity",
    "avg_chloride"
]

df4 = df[cols].dropna()

print("\nLevel 4 Dataset:")
print(df4)

print("\nNumber of rows used:")
print(len(df4))

X = df4[
    [
        "avg_temperature",
        "avg_density",
        "avg_viscosity",
        "avg_chloride"
    ]
]

y = df4["emulsion_level_m"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

a = model.intercept_

b1 = model.coef_[0]
b2 = model.coef_[1]
b3 = model.coef_[2]
b4 = model.coef_[3]

print("\nLEVEL 4 MODEL")

print(
    f"E = {a:.4f} + "
    f"({b1:.4f})*T + "
    f"({b2:.4f})*D + "
    f"({b3:.4f})*V + "
    f"({b4:.8f})*Cl"
)

print(f"\nR² = {r2:.4f}")

print("\nMASTER DATASET INFO")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for c in df.columns:
    print(c)

    # ==========================
# LINEAR REGRESSION MODEL
# ==========================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

features = [
    "avg_temperature",
    "avg_density",
    "avg_viscosity",
    "avg_chloride"
]

target = "emulsion_level_m"

df_lr = df[[target] + features].dropna()

X = df_lr[features]
y = df_lr[target]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

rmse = mean_squared_error(
    y,
    y_pred
) ** 0.5

print("\nLINEAR REGRESSION RESULTS")
print("Rows Used:", len(df_lr))

print("\nR² =", round(r2,4))
print("RMSE =", round(rmse,4))
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

plt.title("Linear Regression: Actual vs Predicted")

plt.show()

print("\nMISSING VALUES PER COLUMN")
print(df.isnull().sum().sort_values(ascending=False))

print("\nRows =", len(df))
print("Columns =", len(df.columns))



