import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


st.set_page_config(page_title="Student Score Predictor", layout="centered")

st.title("📘 Student Exam Score Prediction")
st.write("Predict final exam score using Linear Regression")


@st.cache_data
def load_data():
    return pd.read_csv("student_exam_score_dataset.csv")

data = load_data()


with st.expander("View Dataset"):
    st.dataframe(data)

X = data[['hours_studied', 'attendance_percent', 'previous_score']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.sidebar.header("Enter Student Details")

hours = st.sidebar.slider("Hours Studied per Day", 0, 12, 5)
attendance = st.sidebar.slider("Attendance Percentage", 50, 100, 75)
previous_score = st.sidebar.slider("Previous Exam Score", 0, 100, 60)

input_data = pd.DataFrame({
    'hours_studied': [hours],
    'attendance_percent': [attendance],
    'previous_score': [previous_score]
})

prediction = model.predict(input_data)[0]

st.subheader("Predicted Final Score")
st.success(f"{prediction:.2f}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

with st.expander("How this works"):
    st.write("""
    - This app uses **Multiple Linear Regression**
    - Inputs: Study Hours, Attendance, Previous Score
    - Output: Predicted Final Exam Score
    - Model evaluated using **MAE** and **R² Score**
    """)
