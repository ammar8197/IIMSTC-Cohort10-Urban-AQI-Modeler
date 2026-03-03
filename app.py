import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join("models", "pm25_xgb_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please check the models folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="PM2.5 Prediction", layout="centered")

st.title("🌫 Urban AQI - PM2.5 Prediction System")
st.markdown("Provide the required environmental inputs below.")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Required Input Features")

col1, col2 = st.columns(2)

with col1:
    PM10 = st.number_input("PM10 (µg/m³)", min_value=0.0)
    SO2 = st.number_input("SO2 (µg/m³)", min_value=0.0)
    NO2 = st.number_input("NO2 (µg/m³)", min_value=0.0)
    CO = st.number_input("CO (mg/m³)", min_value=0.0)
    O3 = st.number_input("O3 (µg/m³)", min_value=0.0)
    TEMP = st.number_input("Temperature (°C)")

with col2:
    PRES = st.number_input("Pressure (hPa)")
    DEWP = st.number_input("Dew Point (°C)")
    RAIN = st.number_input("Rainfall (mm)", min_value=0.0)
    WSPM = st.number_input("Wind Speed (m/s)", min_value=0.0)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, step=1)
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, step=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict PM2.5"):

    input_data = np.array([[PM10, SO2, NO2, CO, O3,
                            TEMP, PRES, DEWP, RAIN, WSPM,
                            month, hour]])

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted PM2.5: {prediction:.2f} µg/m³")

        # AQI Interpretation
        if prediction <= 50:
            st.info("Air Quality: Good")
        elif prediction <= 100:
            st.warning("Air Quality: Moderate")
        else:
            st.error("Air Quality: Unhealthy")

    except Exception as e:
        st.error(f"Prediction failed: {e}")