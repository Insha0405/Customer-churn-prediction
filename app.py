import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("customer_churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Customer Churn Prediction System")

st.markdown("""
Welcome to the **AI-Powered Customer Churn Prediction System**.

This application predicts whether a telecom customer is likely to **leave (churn)** or **stay** with the company based on customer profile and billing information.

### 📌 What is Customer Churn?

Customer churn refers to customers who discontinue a company's service. Predicting churn helps businesses retain valuable customers through proactive engagement.
""")

st.divider()

st.subheader("Customer Details")

st.subheader("👤 Customer Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Select the customer's gender."
)

senior = st.selectbox(
    "Senior Citizen",
    [0,1],
    format_func=lambda x: "Yes" if x==1 else "No",
    help="Is the customer 65 years or older?"
)

partner = st.selectbox(
    "Has a Partner?",
    ["Yes","No"],
    help="Whether the customer has a spouse or partner."
)

dependents = st.selectbox(
    "Has Dependents?",
    ["Yes","No"],
    help="Whether the customer has children or other dependents."
)

st.divider()

st.subheader("💳 Billing Information")

tenure = st.slider(
    "Customer Tenure (Months)",
    1,
    72,
    12,
    help="Number of months the customer has been with the company."
)

monthly = st.number_input(
    "Monthly Charges ($)",
    min_value=18.25,
    max_value=120.0,
    value=70.0,
    help="Average monthly bill paid by the customer."
)

total = st.number_input(
    "Total Charges ($)",
    min_value=18.25,
    max_value=9000.0,
    value=1000.0,
    help="Total amount paid by the customer so far."
)
if st.button("🔍 Predict Customer Churn", use_container_width=True):

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

    st.divider()

st.subheader("📈 Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ High Risk of Customer Churn")

    st.markdown("""
### Recommended Business Actions

- 📞 Contact the customer with a personalized retention offer.
- 💰 Provide discounts or loyalty rewards.
- 📄 Recommend switching to a long-term contract.
- 🤝 Schedule a customer success follow-up.
""")

else:
    st.success("✅ Customer is Likely to Stay")

    st.markdown("""
### Recommendation

- 😊 Continue providing quality service.
- 🎁 Offer loyalty benefits to strengthen customer satisfaction.
""")        