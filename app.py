import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Anxiety Level Predictor",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
    model = joblib.load(BASE_DIR / "best_anxiety_model.pkl")
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error("Could not load the trained model artifacts.")
    st.code(str(e))
    st.stop()

st.title("🧠 Anxiety Level Prediction")
st.caption("Machine-learning prediction of an anxiety score from 1 to 10.")

st.info(
    "This is an educational machine-learning demonstration. "
    "It is not a medical diagnosis or a substitute for professional care."
)

with st.form("anxiety_prediction_form"):
    st.subheader("Personal & lifestyle information")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        occupation = st.selectbox(
            "Occupation",
            ["Artist", "Chef", "Doctor", "Engineer", "Freelancer",
             "Lawyer", "Musician", "Nurse", "Other", "Scientist",
             "Student", "Teacher"]
        )
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)

    with c2:
        physical_activity = st.slider(
            "Physical Activity (hrs/week)", 0.0, 15.0, 4.0, 0.1
        )
        caffeine = st.number_input(
            "Caffeine Intake (mg/day)", min_value=0, max_value=1000, value=200, step=10
        )
        alcohol = st.number_input(
            "Alcohol Consumption (drinks/week)", min_value=0, max_value=30, value=4, step=1
        )
        smoking = st.selectbox("Smoking", ["No", "Yes"])

    with c3:
        family_history = st.selectbox("Family History of Anxiety", ["No", "Yes"])
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        heart_rate = st.number_input(
            "Heart Rate (bpm)", min_value=40, max_value=180, value=80, step=1
        )
        breathing_rate = st.number_input(
            "Breathing Rate (breaths/min)", min_value=8, max_value=40, value=18, step=1
        )

    st.subheader("Symptoms, treatment & recent factors")

    c4, c5, c6 = st.columns(3)

    with c4:
        sweating = st.slider("Sweating Level (1-5)", 1, 5, 3)
        dizziness = st.selectbox("Dizziness", ["No", "Yes"])
        medication = st.selectbox("Medication", ["No", "Yes"])

    with c5:
        therapy = st.number_input(
            "Therapy Sessions (per month)", min_value=0, max_value=20, value=1, step=1
        )
        major_event = st.selectbox("Recent Major Life Event", ["No", "Yes"])
        diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 5)

    with c6:
        st.markdown("### Prediction")
        submitted = st.form_submit_button(
            "🔮 Predict Anxiety Level", use_container_width=True
        )

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Sleep Hours": sleep_hours,
        "Physical Activity (hrs/week)": physical_activity,
        "Caffeine Intake (mg/day)": caffeine,
        "Alcohol Consumption (drinks/week)": alcohol,
        "Smoking": smoking,
        "Family History of Anxiety": family_history,
        "Stress Level (1-10)": stress,
        "Heart Rate (bpm)": heart_rate,
        "Breathing Rate (breaths/min)": breathing_rate,
        "Sweating Level (1-5)": sweating,
        "Dizziness": dizziness,
        "Medication": medication,
        "Therapy Sessions (per month)": therapy,
        "Recent Major Life Event": major_event,
        "Diet Quality (1-10)": diet_quality,
    }])

    try:
        transformed = preprocessor.transform(input_data)
        raw_prediction = float(model.predict(transformed)[0])
        prediction = float(np.clip(raw_prediction, 1, 10))
        rounded_prediction = int(np.clip(np.rint(prediction), 1, 10))

        if rounded_prediction <= 3:
            band = "Lower predicted anxiety level"
        elif rounded_prediction <= 6:
            band = "Moderate predicted anxiety level"
        else:
            band = "Higher predicted anxiety level"

        st.success("Prediction completed.")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Predicted Anxiety Level", f"{rounded_prediction}/10")
        with m2:
            st.metric("Model Score", f"{prediction:.2f}/10")

        st.progress(rounded_prediction / 10)
        st.write(f"**Interpretation:** {band}.")

        st.caption(
            "The score is a model prediction based on the patterns present in "
            "the supplied dataset. It should not be interpreted as a clinical assessment."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

with st.expander("About this project"):
    st.write(
        "The model was trained on the supplied enhanced anxiety dataset. "
        "Categorical variables are one-hot encoded and numeric variables are "
        "passed through to a Random Forest regression model."
    )
    st.write(
        "The target variable is **Anxiety Level (1-10)**."
    )
