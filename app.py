import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Athlete Fatigue Risk Prediction",
    page_icon="🏃",
    layout="centered"
)

model = joblib.load("fatigue_risk_model.pkl")

st.title("🏃 Athlete Fatigue Risk Detection")
st.markdown(
    "Enter the athlete's training and physiological metrics to predict fatigue risk."
)

training_hours = st.number_input(
    "Training Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5
)

training_intensity = st.slider(
    "Training Intensity", min_value=1, max_value=10, value=5
)

heart_rate_recovery = st.number_input(
    "Heart Rate Recovery", min_value=0.0, value=25.0
)

lactate_level = st.number_input(
    "Cortisol Level", min_value=0.0, value=15.0
)

rpe = st.slider(
    "RPE (Rate of Perceived Exertion)", min_value=1, max_value=10, value=5
)

sleep_hours = st.number_input(
    "Sleep Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5
)

sleep_quality = st.slider(
    "Sleep Quality", min_value=1, max_value=10, value=7
)

muscle_soreness = st.slider(
    "Muscle Soreness", min_value=1, max_value=10, value=3
)

hrv = st.number_input(
    "HRV", min_value=0.0, value=60.0
)

if st.button("Predict Fatigue Risk"):

    input_df = pd.DataFrame({
        "HRV": [hrv],
        "HeartRate_Recovery": [heart_rate_recovery],
        "Cortisol_Level": [lactate_level],
        "Sleep_Quality": [sleep_quality],
        "Sleep_Hours": [sleep_hours],
        "Muscle_Soreness": [muscle_soreness],
        "RPE": [rpe],
        "Training_Hours": [training_hours],
        "Training_Intensity": [training_intensity]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Fatigue Risk")
    else:
        st.success("✅ Low Fatigue Risk")

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(input_df).max()
        st.write(f"**Prediction Confidence:** {confidence:.2%}")
