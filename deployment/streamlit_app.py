import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="PM2.5 Air Quality Predictor",
    layout="wide"
)
st.title("PM2.5 Air Quality Predictor")
st.markdown("Predict PM2.5 concentration levels based on air quality parameters")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_pm25_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("✓ Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Input Parameters")
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    pm10 = st.sidebar.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.2f")
    so2 = st.sidebar.number_input("SO2 (ppb)", min_value=0.0, max_value=500.0, value=10.0, step=0.1, format="%.2f")
    no2 = st.sidebar.number_input("NO2 (ppb)", min_value=0.0, max_value=300.0, value=45.0, step=0.1, format="%.2f")
    co = st.sidebar.number_input("CO (ppm)", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f")
    o3 = st.sidebar.number_input("O3 (ppb)", min_value=0.0, max_value=250.0, value=50.0, step=0.1, format="%.2f")
    temp = st.sidebar.number_input("Temperature (°C)", min_value=-40.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")

with col2:
    pres = st.sidebar.number_input("Pressure (hPa)", min_value=900.0, max_value=1050.0, value=1013.0, step=0.1, format="%.1f")
    dewp = st.sidebar.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, value=15.0, step=0.1, format="%.1f")
    rain = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, format="%.1f")
    wspm = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.0, step=0.1, format="%.1f")
    month = st.sidebar.selectbox("Month", list(range(1,13)), index=2)
    hour = st.sidebar.number_input("Hour (0-23)", min_value=0, max_value=23, value=14, step=1, format="%d")

features = np.array([[pm10, so2, no2, co, o3, temp, pres, dewp, rain, wspm, month, hour]])

prediction = model.predict(features)[0]

st.markdown("---")

col_metric1, col_metric2 = st.columns([2, 1])

with col_metric1:
    st.metric(
        label="Predicted PM2.5 Concentration",
        value=f"{prediction:.2f} µg/m³",
        delta=None,
        delta_color="normal"
    )

def get_aqi_status(pm25):
    if pm25 <= 35:
        status = "GOOD"
        color = "#00b050"
        advice = "Air quality is satisfactory. Enjoy outdoor activities!"
    elif pm25 <= 75:
        status = "MODERATE"
        color = "#ffc000"
        advice = "Air quality is acceptable. Sensitive groups may experience issues."
    elif pm25 <= 115:
        status = "UNHEALTHY FOR SENSITIVE GROUPS"
        color = "#ff6600"
        advice = "Sensitive groups should limit outdoor activities."
    elif pm25 <= 150:
        status = "UNHEALTHY"
        color = "#ff0000"
        advice = "Everyone may begin to experience health effects. Limit outdoor activities."
    else:
        status = "VERY UNHEALTHY"
        color = "#8b0000"
        advice = "Health alert! Avoid outdoor activities."
    
    return status, color, advice

status, color, advice = get_aqi_status(prediction)

with col_metric2:
    st.markdown(f"<h3 style='color: {color}'>{status}</h3>", unsafe_allow_html=True)

st.info(f"**Advisory:** {advice}")

st.markdown("---")
st.subheader("Input Summary")

input_data = {
    "Parameter": [
        "PM10", "SO2", "NO2", "CO", "O3", 
        "Temperature", "Pressure", "Dew Point", "Rainfall", "Wind Speed", 
        "Month", "Hour"
    ],
    "Value": [
        pm10, so2, no2, co, o3, 
        temp, pres, dewp, rain, wspm, 
        month, hour
    ],
    "Unit": [
        "µg/m³", "ppb", "ppb", "ppm", "ppb",
        "°C", "hPa", "°C", "mm", "m/s",
        "", ""
    ]
}

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Pollutants**")
    st.write(f"PM10: {pm10} µg/m³")
    st.write(f"SO2: {so2} ppb")
    st.write(f"NO2: {no2} ppb")
    st.write(f"CO: {co} ppm")
    st.write(f"O3: {o3} ppb")

with col2:
    st.write("**Weather**")
    st.write(f"Temperature: {temp}°C")
    st.write(f"Pressure: {pres} hPa")
    st.write(f"Dew Point: {dewp}°C")
    st.write(f"Rainfall: {rain} mm")
    st.write(f"Wind Speed: {wspm} m/s")

with col3:
    st.write("**Time**")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.write(f"Month: {months[month-1]}")
    st.write(f"Hour: {hour:02d}:00")

st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 12px; color: gray;'>
    IIMSTC Cohort 10 - Urban AQI Modeler Project
    </p>
</div>
""", unsafe_allow_html=True)
