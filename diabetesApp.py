import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns

st.title("Diabetes Predictor App")
st.write("From the diabetes data, we built a machine learning model for diabetes predictions.")

st.sidebar.title("Diabetes Predictor App Parameters")
st.sidebar.write("Tweak to change predictions")

# Glucose
glucose = st.sidebar.slider("Glucose", 0, 200, 70)

# BMI
bmi = st.sidebar.slider("BMI", 0.0, 100.9, 50.0)

# Age
age = st.sidebar.slider("Age", 0, 150, 15)

# Blood Pressure
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 300, 100)


# Main Page
st.subheader("Predictions")

# Loading the model
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

# [Glucose, BMI, Age, BloodPressure]
prediction = round(loaded_model.predict([[glucose, bmi, age, blood_pressure]])[0])

if prediction == 0:
    risk_status = "No"
else:
    risk_status = "Yes"

st.write(f"Risk to Diabetes?: {risk_status}")

# Load data
data = pd.read_csv("diabetes_classification.csv")

if st.checkbox("Show Graphs"):
    sns.pairplot(data[['Glucose', 'BMI', 'Age', 'BloodPressure']], height=8, kind='reg', diag_kind='kde')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
