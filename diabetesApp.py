import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import time


@st.cache(suppress_st_warning=True)
def load_csv_data(_dir, head=0, tail=0):
    file = pd.read_csv(_dir)
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Loading CSV Data... {i+1}%')
        bar.progress(i+1)
        time.sleep(0.01)
    bar.empty()
    latest_iteration.text('')
    if head > 0 and tail == 0:
        return file.head(head)
    elif head == 0 and tail > 0:
        return file.head(tail)
    return file


def load_chart(data, kind):
    if kind == 'line':
        st.write("Line Chart")
        st.line_chart(data)
    elif kind == 'area':
        st.write("Area Chart")
        st.area_chart(data)
    elif kind == 'bar':
        st.write("Bar Chart")
        st.bar_chart(data)
    else:
        st.write("Line Chart")
        st.line_chart(data)
        

#st.set_page_config(layout="wide")
st.title("Diabetes Predictor App")
st.write("From the diabetes data, we built a machine learning model for diabetes predictions.")

# Initialize CSV data
filename = "diabetes_classification.csv"
file = load_csv_data(filename, head=50)

# Initialize columns and target
columns = ['Glucose', 'BMI', 'Age', 'BloodPressure']
target = 'Outcome'

# Sidebar
st.sidebar.title("Diabetes Predictor App Parameters")
st.sidebar.write("Tweak to change predictions")

# Glucose
glucose = st.sidebar.slider("Glucose", 0, 200, 70)

# BMI

bmi = st.sidebar.slider("BMI", 0.0, 100, 23)

# Age
age = st.sidebar.slider("Age", 0, 150, 28)

# Blood Pressure
bloodpressure = st.sidebar.slider("Blood Pressure", 0, 250, 70)

# Dataframe visibility
st.sidebar.subheader("Data Frame Visibility")
option_sidebar = st.sidebar.checkbox("Hide")
if not option_sidebar:
    st.caption(f"Data Frame: '{filename}'")
    st.write(file)
    st.write("\n\n")
    
# Line chart
load_chart(file[columns], "line")

# Area chart
load_chart(file[columns], "area")

# Bar chart
load_chart(file[columns], "bar")

# Main Page
st.subheader("Predictions")

# Loading the model
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

# [Glucose, BMI, Age, BloodPressure]
prediction = round(loaded_model.predict([[glucose, bmi, age, bloodpressure]])[0])

if prediction == 0:
    risk_status = "No"
else:
    risk_status = "Yes"

st.write(f"Risk to Diabetes?: {risk_status}")

# Load data
data = pd.read_csv("diabetes_classification.csv")

if st.checkbox("Show Graphs"):
    sns.pairplot(data[['Glucose', 'BMI', 'Age', 'BloodPressure']], height=8, diag_kind='kde', hue='Glucose')
  
   
    st.set_option('deprecation.showPyplotGlobalUse', False)
   
    
    st.pyplot()
    
