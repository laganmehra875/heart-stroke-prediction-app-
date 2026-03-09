import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Lalit ❤️")
st.markdown("Provide the following details")

# User Inputs
Age = st.slider("Age", 18, 100, 40)
Sex = st.selectbox("Sex", ["M", "F"])
Chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

Resting_BP = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

FastingBS = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

MaxHR = st.slider("Max Heart Rate", 60, 220, 150)

ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

Oldpeak = st.slider("OldPeak (ST Depression)", 0.0, 6.0, 1.0)

ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# Prediction Button
if st.button("Predict"):

    raw_input = {
        "Age": Age,
        "RestingBP": Resting_BP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "MaxHR": MaxHR,
        "Oldpeak": Oldpeak,
        "Sex_" + Sex: 1,
        "ChestPainType_" + Chest_pain: 1,
        "RestingECG_" + RestingECG: 1,
        "ExerciseAngina_" + ExerciseAngina: 1,
        "ST_Slope_" + ST_Slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale data
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")