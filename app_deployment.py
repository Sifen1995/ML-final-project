import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and tools
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title("🎓 Student Performance Predictor")
st.write("Enter student details below to predict the final G3 grade.")

# Create User Inputs based on key features
st.sidebar.header("Input Student Data")

absences = st.sidebar.slider("Absences", 0, 93, 0)
studytime = st.sidebar.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                                 help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
failures = st.sidebar.selectbox("Previous Class Failures", [0, 1, 2, 3, 4])
g1 = st.sidebar.number_input("G1 Grade (0-20)", 0, 20, 10)
g2 = st.sidebar.number_input("G2 Grade (0-20)", 0, 20, 10)

# Prepare the data for prediction
# For deployment simplicity, we fill other features with 0/median 
# or you can add more inputs here
input_data = pd.DataFrame(0, index=[0], columns=features)
input_data['absences'] = absences
input_data['studytime'] = studytime
input_data['failures'] = failures
input_data['G1'] = g1
input_data['G2'] = g2

# Scale and Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Display Results
st.subheader(f"Predicted Final Grade (G3): {prediction:.2f}")

if prediction >= 10:
    st.success("Result: Student is likely to PASS.")
else:
    st.error("Result: Student is at risk of FAILING.")