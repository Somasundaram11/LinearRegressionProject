import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Student Score Predictor")

st.title("🎓 Student Final Score Prediction")
st.write("Predict final exam score based on study behavior")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model, scaler, imputer = pickle.load(f)

# User inputs
hours = st.number_input("📘 Hours Studied", min_value=0.0, max_value=24.0)
attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0)
previous = st.number_input("📝 Previous Score", min_value=0.0, max_value=100.0)

# Prediction
if st.button("Predict Final Score"):
    X = np.array([[hours, attendance, previous]])
    X = imputer.transform(X)
    X = scaler.transform(X)

    prediction = model.predict(X)
    st.success(f"✅ Predicted Final Score: {prediction[0]:.2f}")
