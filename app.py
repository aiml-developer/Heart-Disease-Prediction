import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model and Scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# Create Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                                  format_func=lambda x: f"{x} - " + ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                   format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=[1, 2, 3], 
                                format_func=lambda x: ["Upward", "Flat", "Downward"][x-1])

    submit_btn = st.form_submit_button("Predict Result")

if submit_btn:
    # Prepare Data for Prediction
    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
    
    # Scale Data
    input_data_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Show Result
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Normal - Low Risk of Heart Disease. (Probability: {probability:.2%})")
