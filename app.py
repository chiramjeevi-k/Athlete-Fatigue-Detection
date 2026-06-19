import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Athlete Fatigue Risk Prediction",
    page_icon="🏃",
    layout="wide"
)

model = joblib.load("fatigue_risk_model.pkl")

with st.sidebar:
    st.title("About Project")

    st.write("""
    Athlete Fatigue Risk Prediction System

    This application uses Machine Learning
    to predict athlete fatigue risk using
    physiological and training metrics.
    """)

    st.markdown("### Model")
    st.write("Logistic Regression")

    st.markdown("### Features")
    st.markdown("""
    - HRV
    - Heart Rate Recovery
    - Cortisol Level
    - Sleep Quality
    - Sleep Hours
    - Muscle Soreness
    - RPE
    - Training Hours
    - Training Intensity
    """)

st.title("🏃 Athlete Fatigue Risk Prediction")

st.markdown(
    "Enter the athlete's physiological and training metrics to predict fatigue risk."
)


st.info("""
**Workflow**

Athlete Data → Preprocessing → Model Inference → Fatigue Risk Prediction
""")

st.markdown("---")

input_col, result_col = st.columns([2, 1])

with input_col:

    training_hours = st.number_input(
        "Training Hours",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    training_intensity = st.slider(
        "Training Intensity",
        min_value=1,
        max_value=10,
        value=5
    )

    heart_rate_recovery = st.number_input(
        "Heart Rate Recovery",
        min_value=0.0,
        value=25.0
    )

    cortisol_level = st.number_input(
        "Cortisol Level",
        min_value=0.0,
        value=15.0
    )

    rpe = st.slider(
        "RPE (Rate of Perceived Exertion)",
        min_value=1,
        max_value=10,
        value=5
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=8.0,
        step=0.5
    )

    sleep_quality = st.slider(
        "Sleep Quality",
        min_value=1,
        max_value=10,
        value=7
    )

    muscle_soreness = st.slider(
        "Muscle Soreness",
        min_value=1,
        max_value=10,
        value=3
    )

    hrv = st.number_input(
        "HRV",
        min_value=0.0,
        value=60.0
    )

predict_button = st.button("Predict Fatigue Risk")

if predict_button:

    input_df = pd.DataFrame({
        "HRV": [hrv],
        "HeartRate_Recovery": [heart_rate_recovery],
        "Cortisol_Level": [cortisol_level],
        "Sleep_Quality": [sleep_quality],
        "Sleep_Hours": [sleep_hours],
        "Muscle_Soreness": [muscle_soreness],
        "RPE": [rpe],
        "Training_Hours": [training_hours],
        "Training_Intensity": [training_intensity]
    })

    prediction = model.predict(input_df)[0]

    with result_col:

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ HIGH FATIGUE RISK")

            st.warning("""
            Recommendations:
            - Increase recovery time
            - Improve sleep quality
            - Reduce training intensity
            - Monitor HRV regularly
            - Ensure proper hydration
            """)

        else:

            st.success("✅ LOW FATIGUE RISK")

            st.info("""
            Athlete appears well recovered.

            Current training load appears sustainable.
            """)

        if hasattr(model, "predict_proba"):

            confidence = model.predict_proba(input_df).max()

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.progress(float(confidence))

st.markdown("---")
st.caption(
    "Athlete Fatigue Risk Detection System"
)