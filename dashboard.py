import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Wash Tank AI Dashboard",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("master_dataset.csv")

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

df_rf = df[[target] + features].dropna()

X = df_rf[features]
y = df_rf[target]

# ==========================
# TRAIN MODEL
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==========================
# DASHBOARD HEADER
# ==========================

st.title("Wash Tank AI Dashboard")
st.header("Thermal Image Analysis")

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Thermal Image",
        width=300
    )

    image_np = np.array(image)

    if len(image_np.shape) == 3:

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_np

    avg_brightness = np.mean(gray)

    min_pixel = np.min(gray)

    max_pixel = np.max(gray)

    thermal_range = max_pixel - min_pixel

    hotspot_pixels = np.sum(
        gray > 220
    )

    total_pixels = gray.size

    hotspot_ratio = (
        hotspot_pixels / total_pixels
    )

    upper_half = gray[
        :gray.shape[0]//2,
        :
    ]

    lower_half = gray[
        gray.shape[0]//2:,
        :
    ]

    upper_avg = np.mean(
        upper_half
    )

    lower_avg = np.mean(
        lower_half
    )

    vertical_gradient = (
        lower_avg - upper_avg
    )

    thermal_uniformity = np.std(
        gray
    )

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Brightness", f"{avg_brightness:.2f}")

with c2:
    st.metric("Thermal Range", f"{thermal_range:.2f}")

with c3:
    st.metric("Hotspot Ratio", f"{hotspot_ratio:.4f}")

with c4:
    st.metric("Vertical Gradient", f"{vertical_gradient:.2f}")

with c5:
    st.metric("Uniformity", f"{thermal_uniformity:.2f}")

# حفظ النتائج بعد الانتهاء من العرض

st.session_state["thermal_range"] = float(
    thermal_range
)

st.session_state["hotspot_ratio"] = float(
    hotspot_ratio
)

st.session_state["vertical_gradient"] = float(
    vertical_gradient
)

st.session_state["thermal_uniformity"] = float(
    thermal_uniformity
)

left_col, center_col, right_col = st.columns([1,1,1.2])

# ==========================
# LEFT COLUMN - INPUTS
# ==========================

with left_col:

    st.header("Inputs")

    temp = st.number_input(
        "Average Temperature",
        value=float(df_rf["avg_temperature"].mean())
    )

    pressure = st.number_input(
        "Average Pressure",
        value=float(df_rf["avg_pressure"].mean())
    )

    level = st.number_input(
        "Average Level",
        value=float(df_rf["avg_level"].mean())
    )

    oil_outlet = st.number_input(
        "Average Oil Outlet",
        value=float(df_rf["avg_oil_outlet"].mean())
    )

    water_outlet = st.number_input(
        "Average Water Outlet",
        value=float(df_rf["avg_water_outlet"].mean())
    )

    dosage = st.number_input(
        "Daily Dosage",
        value=float(df_rf["daily_dosage"].mean())
    )

    clean_water = st.number_input(
        "Clean Water Points",
        value=float(df_rf["clean_water_points"].mean())
    )

    emulsion_points = st.number_input(
        "Emulsion Points",
        value=float(df_rf["emulsion_points"].mean())
    )

    oil_points = st.number_input(
        "Oil Points",
        value=float(df_rf["oil_points"].mean())
    )

    thermal_range = st.number_input(
        "Thermal Range",
        value=float(
            st.session_state.get(
                "thermal_range",
                df_rf["avg_thermal_range"].mean()
            )
        )
    )

    hotspot_ratio = st.number_input(
        "Hotspot Ratio",
        value=float(
            st.session_state.get(
                "hotspot_ratio",
                df_rf["avg_hotspot_ratio"].mean()
            )
        )
    )

    vertical_gradient = st.number_input(
        "Vertical Gradient",
        value=float(
            st.session_state.get(
                "vertical_gradient",
                df_rf["avg_vertical_gradient"].mean()
            )
        )
    )

    thermal_uniformity = st.number_input(
        "Thermal Uniformity",
        value=float(
            st.session_state.get(
                "thermal_uniformity",
                df_rf["avg_thermal_uniformity"].mean()
            )
        )
    )

    predict_button = st.button("Predict")

# ==========================
# CALCULATE PREDICTION
# ==========================

prediction = None

if predict_button:

    new_data = pd.DataFrame([[
        temp,
        pressure,
        level,
        oil_outlet,
        water_outlet,
        dosage,
        clean_water,
        emulsion_points,
        oil_points,
        thermal_range,
        hotspot_ratio,
        vertical_gradient,
        thermal_uniformity
    ]], columns=features)

prediction = model.predict(new_data)[0]

# حساب الثقة
tree_predictions = []

for tree in model.estimators_:
    tree_predictions.append(
        tree.predict(new_data)[0]
    )

std_dev = np.std(tree_predictions)

confidence = max(
    0,
    min(
        100,
        100 - (std_dev * 20)
    )
)

# حساب الطبقات
level_m = level / 1000

interface_height = prediction

oil_height = level_m - interface_height

if oil_height < 0:
    oil_height = 0

    if oil_height < 0:
        oil_height = 0

# ==========================
# CENTER COLUMN
# ==========================
with center_col:
    st.header("Prediction")

    if prediction is not None:

        st.success(
            f"Predicted Emulsion Level = {prediction:.2f} m"
        )

        st.info(
            f"Prediction Confidence = {confidence:.1f}%"
        )

        st.info(
            f"Estimated Oil Layer = {oil_height:.2f} m"
        )

# ==========================
# TRAFFIC LIGHT
# ==========================

if prediction < 5.5:
    st.success("🟢 Stable Operation")

elif prediction < 7.0:
    st.warning("🟡 Attention Required")

else:
    st.error("🔴 High Emulsion Risk")

# ==========================
# ROOT CAUSE
# ==========================

st.subheader("Possible Root Cause")

root_causes = []

if oil_outlet > 43:
    root_causes.append(
        "High oil outlet rate may be reducing separation residence time."
    )

if water_outlet > 60:
    root_causes.append(
        "High water outlet rate may be disturbing the interface zone."
    )

if level > 11000:
    root_causes.append(
        "High tank level may reduce settling efficiency."
    )

if temp < 61:
    root_causes.append(
        "Low temperature may increase oil viscosity and stabilize emulsion."
    )

if len(root_causes) == 0:
    st.success(
        "No significant operational cause detected."
    )

else:
    for cause in root_causes:
        st.warning(cause)

# ==========================
# RECOMMENDED ACTIONS
# ==========================

st.subheader("Recommended Actions")

actions = []

if oil_outlet > 43:
    actions.append(
        "Review oil outlet flow rate."
    )

if water_outlet > 60:
    actions.append(
        "Reduce water outlet withdrawal if operationally feasible."
    )

if level > 11000:
    actions.append(
        "Review tank inventory and interface control."
    )

if temp < 61:
    actions.append(
        "Verify heater performance and operating temperature."
    )

if len(actions) == 0:
    st.info(
        "Current operating conditions appear acceptable."
    )

else:
    for action in actions:
        st.info(action)

        if prediction > 7:
            st.warning(
                "High emulsion level detected. Consider increasing chemical dosage."
            )

        elif prediction > 6:
            st.warning(
                "Moderate emulsion level. Monitor tank performance closely."
            )

        else:
            st.success(
                "Emulsion level is within normal operating range."
            )

            st.markdown("---")

# ==========================
# RIGHT COLUMN
# ==========================
with right_col:

    st.header("Tank Simulation")

    if prediction is not None:

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle

        # =====================
        # أبعاد الخزان الحقيقية
        # =====================

        tank_diameter = 30.0
        total_height = 14.6

        roof_height = 2.5
        shell_height = total_height - roof_height

        # =====================
        # إنشاء الشكل
        # =====================

        fig, ax = plt.subplots(figsize=(5.5,3.2))
        ax.set_aspect('equal')

        # =====================
        # جسم الخزان
        # =====================

        ax.plot(
        [0,0],
        [0,shell_height],
        color="black",
        linewidth=3
        )

        ax.plot(
        [tank_diameter,tank_diameter],
        [0,shell_height],
        color="black",
        linewidth=3
        )

        ax.plot(
        [0,tank_diameter],
        [0,0],
        color="black",
        linewidth=3
        )

        ax.plot(
        [0,tank_diameter],
        [shell_height,shell_height],
        color="black",
        linewidth=3
        )

        # =====================
        # السقف المقبب الحقيقي
        # =====================

        theta = np.linspace(0, np.pi, 200)

        x_roof = (
        tank_diameter/2
        +
        (tank_diameter/2) * np.cos(theta)
        )

        y_roof = (
        shell_height
        +
        roof_height * np.sin(theta)
        )

        ax.plot(
        x_roof,
        y_roof,
        color="black",
        linewidth=3
        )

        # =====================
        # طبقة النفط
        # =====================

        oil_rect = Rectangle(
        (0, interface_height),
        tank_diameter,
        oil_height,
        facecolor="black"
        )

        ax.add_patch(oil_rect)

        # =====================
        # طبقة الماء / الإيمولشن
        # =====================

        lower_rect = Rectangle(
        (0,0),
        tank_diameter,
        interface_height,
        facecolor="#cfa86a"
        )

        ax.add_patch(lower_rect)

        # =====================
        # خط الواجهة
        # =====================

        ax.hlines(
        interface_height,
        0,
        tank_diameter,
        colors="yellow",
        linestyles="dashed",
        linewidth=3
        )

        # =====================
        # النصوص
        # =====================

        ax.text(
        tank_diameter/2,
        interface_height/2,
        f"Interface\n{interface_height:.2f} m",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold"
        )

        ax.text(
        tank_diameter/2,
        interface_height + oil_height/2,
        f"Oil\n{oil_height:.2f} m",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold"
        )

        # =====================
        # المحاور
        # =====================

        ax.set_xlim(-1,31)

        ax.set_ylim(
        0,
        total_height + 0.5
        )

        ax.set_aspect("equal")

        ax.set_ylabel("Height (m)")

        ax.set_title("Wash Tank")

        st.pyplot(fig)

        st.info(
                    f"Total Liquid Level = {level_m:.2f} m"
                )

        st.warning(
                    f"Emulsion Interface = {interface_height:.2f} m"
                )

        st.success(
                    f"Oil Layer = {oil_height:.2f} m"
        )
 # ==========================
# AI RECOMMENDATIONS
# ==========================

st.markdown("---")
st.header("AI Recommendations")

if prediction is not None:

    if prediction > 7:

        st.error("""
• High emulsion interface detected

• Increase chemical dosage

• Check water draw-off performance

• Verify pressure stability
""")

    elif prediction > 6:

        st.warning("""
• Moderate emulsion level

• Monitor outlet performance

• Review dosage trend
""")

    else:

        st.success("""
• Tank operation is stable

• No dosage adjustment required

• Continue routine monitoring
""")
# ==========================
# OPERATIONAL DRIVERS
# ==========================

st.markdown("---")
st.header("Operational Drivers")

importance_df = pd.DataFrame({
    "Variable": [
        "Temperature",
        "Pressure",
        "Level",
        "Oil Outlet",
        "Water Outlet",
        "Dosage",
        "Clean Water Points",
        "Emulsion Points",
        "Oil Points",
        "Thermal Range",
        "Hotspot Ratio",
        "Vertical Gradient",
        "Thermal Uniformity"
    ],
    "Importance": model.feature_importances_ * 100
})

# استخراج أهمية المتغيرات من النموذج

drivers = pd.DataFrame({
    "Variable":[
        "Temperature",
        "Pressure",
        "Level",
        "Oil Outlet",
        "Water Outlet",
        "Dosage",
        "Density (Future)",
        "Viscosity (Future)",
        "Salinity (Future)"
    ],
    "Importance":[
        importance_df.loc[
            importance_df["Variable"]=="Temperature",
            "Importance"
        ].values[0],

        importance_df.loc[
            importance_df["Variable"]=="Pressure",
            "Importance"
        ].values[0],

        importance_df.loc[
            importance_df["Variable"]=="Level",
            "Importance"
        ].values[0],

        importance_df.loc[
            importance_df["Variable"]=="Oil Outlet",
            "Importance"
        ].values[0],

        importance_df.loc[
            importance_df["Variable"]=="Water Outlet",
            "Importance"
        ].values[0],

        importance_df.loc[
            importance_df["Variable"]=="Dosage",
            "Importance"
        ].values[0],

        0,
        0,
        0
    ]
})

st.bar_chart(
    drivers.set_index("Variable")
)