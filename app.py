import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('lung_model.pkl', 'rb'))

# Sidebar Instructions
st.sidebar.title("📝 Instructions")
st.sidebar.markdown("""
- Fill all the parameters correctly based on the patient's health.
- Click the **Predict** button to check the risk.
- Gender should be selected as **Male** or **Female**.
- Units are important, refer to info in the expandable section.
""")

st.sidebar.markdown("---")
st.sidebar.title("🛠 Tech Stack")
st.sidebar.markdown("""
- Python
- Streamlit
- Scikit-learn
- Pandas, Numpy
""")

# Title
st.title("🫁 Lung Disease Risk Predictor")

# Parameter Info Section
with st.expander("ℹ️ Click to understand the meaning of each parameter"):
    st.markdown("""
    - **Age**: Patient's age in years.
    - **Gender**: Male or Female.
    - **BMI (kg/m2)**: Body Mass Index, calculated as weight (kg) / height² (m²).
    - **Height (m)**: Patient's height in meters.
    - **History of Heart Failure**: 1 if patient has heart failure, else 0.
    - **Working Place**: 1 if exposed to pollutants at work, else 0.
    - **mMRC**: Modified Medical Research Council Dyspnea Scale (0 to 4).
    - **Smoking Status**: 0 = Never, 1 = Ex-smoker, 2 = Current smoker.
    - **Pack History**: Number of cigarette packs smoked per year.
    - **Vaccination**: 1 if vaccinated, else 0.
    - **Depression**: 1 if diagnosed with depression, else 0.
    - **Dependent**: 1 if the patient depends on others for daily tasks.
    - **Temperature (°C)**: Current body temperature.
    - **Respiratory Rate**: Breaths per minute.
    - **Heart Rate**: Beats per minute.
    - **Blood Pressure**: Systolic value only.
    - **Oxygen Saturation (%)**: Level of O₂ in blood.
    - **Sputum**: 1 if cough produces sputum, else 0.
    - **FEV1**: Forced Expiratory Volume in 1 second.
    """)

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.radio("Gender", options=["Male", "Female"])
bmi = st.number_input("BMI (kg/m2)", min_value=10.0, max_value=50.0)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5)
heart_failure = st.selectbox("History of Heart Failure", [0, 1])
working_place = st.selectbox("Working Place (Pollutant Exposure)", [0, 1])
mmrc = st.slider("mMRC Scale (0-4)", 0, 4)
smoking_status = st.selectbox("Smoking Status", ["Never", "Ex-smoker", "Current smoker"])
pack_history = st.number_input("Pack History", min_value=0.0, max_value=100.0)
vaccination = st.selectbox("Vaccination Done", [0, 1])
depression = st.selectbox("Depression", [0, 1])
dependent = st.selectbox("Dependent on others", [0, 1])
temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0)
resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=10, max_value=50)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200)
bp = st.number_input("Blood Pressure (systolic)", min_value=80, max_value=200)
oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100)
sputum = st.selectbox("Sputum Production", [0, 1])
fev1 = st.number_input("FEV1", min_value=0.0, max_value=5.0)
# copd_gold = st.slider("COPD GOLD Stage (1–4)", 1, 4)

# Encode Gender
gender_encoded = 1 if gender == "Male" else 0

# Encode Smoking Status
smoke_map = {"Never": 0, "Ex-smoker": 1, "Current smoker": 2}
smoking_encoded = smoke_map[smoking_status]

# Collecting the features
features = np.array([[
    age, gender_encoded, bmi, height, heart_failure, working_place,
    mmrc, smoking_encoded, pack_history, vaccination, depression, dependent,
    temperature, resp_rate, heart_rate, bp, oxygen_saturation,
    sputum, fev1
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Lung Disease Detected.")
    else:
        st.success("✅ Low Risk of Lung Disease Detected.")
