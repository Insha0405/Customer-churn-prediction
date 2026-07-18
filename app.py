import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("customer_churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 AI-Powered Customer Churn Prediction")

st.write("Enter customer details below to predict whether the customer is likely to churn.")

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0,1])

partner = st.selectbox("Partner", ["Yes","No"])

dependents = st.selectbox("Dependents", ["Yes","No"])

tenure = st.slider("Tenure (Months)",1,72,12)

monthly = st.number_input("Monthly Charges",18.25,120.0,70.0)

total = st.number_input("Total Charges",18.25,9000.0,1000.0)

if st.button("Predict Churn"):

    input_data = pd.DataFrame(0,index=[0],columns=feature_names)

    input_data["SeniorCitizen"] = senior
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    if "gender_Male" in feature_names:
        input_data["gender_Male"] = 1 if gender=="Male" else 0

    if "Partner_Yes" in feature_names:
        input_data["Partner_Yes"] = 1 if partner=="Yes" else 0

    if "Dependents_Yes" in feature_names:
        input_data["Dependents_Yes"] = 1 if dependents=="Yes" else 0

    prediction = model.predict(input_data)

    if prediction[0]==1:
        st.error("⚠️ Customer is likely to Churn.")
    else:
        st.success("✅ Customer is likely to Stay.")
        