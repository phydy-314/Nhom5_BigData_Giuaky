import pickle
import streamlit as st
import numpy as np

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Health Insurance Prediction")

# Ví dụ input (đổi theo feature thật của bạn)
age = st.number_input("Age", 0, 100)
bmi = st.number_input("BMI", 0.0, 60.0)
children = st.number_input("Children", 0, 10)
smoker = st.selectbox("Smoker", [0, 1])

if st.button("Predict"):
    X = np.array([[age, bmi, children, smoker]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.write("Prediction:", pred)
    st.write("Probability:", round(prob, 3))
