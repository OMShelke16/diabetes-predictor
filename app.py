import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and feature list
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Provide patient details to predict diabetes risk:")

# Input form
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
smoking_history = st.selectbox("Smoking History", [
    "never", "former", "current", "not current", "ever", "no info"
])
bmi = st.number_input("BMI", 10.0, 60.0, step=0.1)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, step=0.1)
glucose = st.number_input("Blood Glucose Level", 50, 500, step=1)

# Build input row
input_dict = {
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'bmi': bmi,
    'HbA1c_level': hba1c,
    'blood_glucose_level': glucose,
    'gender_Female': 1 if gender == "Female" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,
    'smoking_history_never': 1 if smoking_history == "never" else 0,
    'smoking_history_former': 1 if smoking_history == "former" else 0,
    'smoking_history_current': 1 if smoking_history == "current" else 0,
    'smoking_history_not current': 1 if smoking_history == "not current" else 0,
    'smoking_history_ever': 1 if smoking_history == "ever" else 0,
    'smoking_history_no info': 1 if smoking_history == "no info" else 0
}

# Create DataFrame from input
input_df = pd.DataFrame([input_dict])

# Add any missing features with 0 and reorder
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Scale
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Diabetes"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: The person is likely to have diabetes.")
    else:
        st.success("✅ Low Risk: The person is unlikely to have diabetes.")
