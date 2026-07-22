import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Model
# ==========================================

model = joblib.load("customer_churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3{
    color:#0F172A;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
}

.stButton>button:hover{
    background:#1D4ED8;
}

[data-testid="stMetricValue"]{
    color:#2563EB;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================

st.title("📊 AI-Powered Customer Churn Prediction")

st.markdown("""
Predict whether a telecom customer is likely to **Stay** or **Churn**
using a trained Machine Learning model.

This project demonstrates an end-to-end Machine Learning workflow including:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Building
- Streamlit Deployment
""")

st.divider()

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("📌 About")

st.sidebar.info("""
**Model Used**

✔ Logistic Regression

**Dataset**

Telco Customer Churn Dataset

**Libraries**

- Streamlit
- Pandas
- Scikit-learn
""")

st.sidebar.success("Developed using Python")

# ==========================================
# Customer Information
# ==========================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        24
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

st.divider()

# ==========================================
# Internet Services
# ==========================================

st.header("🌐 Internet Services")

col3, col4 = st.columns(2)

with col3:

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

with col4:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

st.divider()

# ==========================================
# Billing Information
# ==========================================

st.header("💳 Billing Information")

col5, col6 = st.columns(2)

with col5:

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=18.25,
        max_value=120.00,
        value=70.00,
        step=0.01
    )

with col6:

    total = st.number_input(
        "Total Charges ($)",
        min_value=18.25,
        max_value=9000.00,
        value=1500.00,
        step=0.01
    )

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.divider()

predict = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)

# ==========================================
# Prepare Input Data
# ==========================================

if predict:

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    # Numerical Features

    input_data["SeniorCitizen"] = senior
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total
        # ==========================================
    # One-Hot Encode Categorical Features
    # ==========================================

    # Gender
    if "gender_Male" in input_data.columns:
        input_data["gender_Male"] = 1 if gender == "Male" else 0

    # Partner
    if "Partner_Yes" in input_data.columns:
        input_data["Partner_Yes"] = 1 if partner == "Yes" else 0

    # Dependents
    if "Dependents_Yes" in input_data.columns:
        input_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0

    # Phone Service
    if "PhoneService_Yes" in input_data.columns:
        input_data["PhoneService_Yes"] = 1 if phone_service == "Yes" else 0

    # Multiple Lines
    if "MultipleLines_Yes" in input_data.columns:
        input_data["MultipleLines_Yes"] = 1 if multiple_lines == "Yes" else 0

    if "MultipleLines_No phone service" in input_data.columns:
        input_data["MultipleLines_No phone service"] = (
            1 if multiple_lines == "No phone service" else 0
        )

    # Internet Service
    if "InternetService_Fiber optic" in input_data.columns:
        input_data["InternetService_Fiber optic"] = (
            1 if internet_service == "Fiber optic" else 0
        )

    if "InternetService_No" in input_data.columns:
        input_data["InternetService_No"] = (
            1 if internet_service == "No" else 0
        )

    # Online Security
    if "OnlineSecurity_Yes" in input_data.columns:
        input_data["OnlineSecurity_Yes"] = (
            1 if online_security == "Yes" else 0
        )

    if "OnlineSecurity_No internet service" in input_data.columns:
        input_data["OnlineSecurity_No internet service"] = (
            1 if online_security == "No internet service" else 0
        )

    # Online Backup
    if "OnlineBackup_Yes" in input_data.columns:
        input_data["OnlineBackup_Yes"] = (
            1 if online_backup == "Yes" else 0
        )

    if "OnlineBackup_No internet service" in input_data.columns:
        input_data["OnlineBackup_No internet service"] = (
            1 if online_backup == "No internet service" else 0
        )

    # Device Protection
    if "DeviceProtection_Yes" in input_data.columns:
        input_data["DeviceProtection_Yes"] = (
            1 if device_protection == "Yes" else 0
        )

    if "DeviceProtection_No internet service" in input_data.columns:
        input_data["DeviceProtection_No internet service"] = (
            1 if device_protection == "No internet service" else 0
        )

    # Tech Support
    if "TechSupport_Yes" in input_data.columns:
        input_data["TechSupport_Yes"] = (
            1 if tech_support == "Yes" else 0
        )

    if "TechSupport_No internet service" in input_data.columns:
        input_data["TechSupport_No internet service"] = (
            1 if tech_support == "No internet service" else 0
        )

    # Streaming TV
    if "StreamingTV_Yes" in input_data.columns:
        input_data["StreamingTV_Yes"] = (
            1 if streaming_tv == "Yes" else 0
        )

    if "StreamingTV_No internet service" in input_data.columns:
        input_data["StreamingTV_No internet service"] = (
            1 if streaming_tv == "No internet service" else 0
        )

    # Streaming Movies
    if "StreamingMovies_Yes" in input_data.columns:
        input_data["StreamingMovies_Yes"] = (
            1 if streaming_movies == "Yes" else 0
        )

    if "StreamingMovies_No internet service" in input_data.columns:
        input_data["StreamingMovies_No internet service"] = (
            1 if streaming_movies == "No internet service" else 0
        )

    # Contract
    if "Contract_One year" in input_data.columns:
        input_data["Contract_One year"] = (
            1 if contract == "One year" else 0
        )

    if "Contract_Two year" in input_data.columns:
        input_data["Contract_Two year"] = (
            1 if contract == "Two year" else 0
        )

    # Paperless Billing
    if "PaperlessBilling_Yes" in input_data.columns:
        input_data["PaperlessBilling_Yes"] = (
            1 if paperless == "Yes" else 0
        )

    # Payment Method
    if "PaymentMethod_Credit card (automatic)" in input_data.columns:
        input_data["PaymentMethod_Credit card (automatic)"] = (
            1 if payment_method == "Credit card (automatic)" else 0
        )

    if "PaymentMethod_Electronic check" in input_data.columns:
        input_data["PaymentMethod_Electronic check"] = (
            1 if payment_method == "Electronic check" else 0
        )

    if "PaymentMethod_Mailed check" in input_data.columns:
        input_data["PaymentMethod_Mailed check"] = (
            1 if payment_method == "Mailed check" else 0
        )

    # ==========================================
    # Model Prediction
    # ==========================================

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
        # ==========================================
    # Prediction Results
    # ==========================================

    st.divider()

    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to Churn")
        else:
            st.success("✅ Customer is likely to Stay")

    with col2:

        st.metric(
            label="Churn Probability",
            value=f"{probability*100:.2f}%"
        )

    # ==========================================
    # Risk Level
    # ==========================================

    st.subheader("📈 Customer Risk Level")

    if probability < 0.30:

        st.success("🟢 LOW RISK")

        st.progress(int(probability*100))

    elif probability < 0.60:

        st.warning("🟡 MEDIUM RISK")

        st.progress(int(probability*100))

    else:

        st.error("🔴 HIGH RISK")

        st.progress(int(probability*100))

    st.divider()

    # ==========================================
    # Business Recommendations
    # ==========================================

    if prediction[0] == 1:

        st.subheader("💡 Recommended Retention Strategy")

        st.markdown("""
### Immediate Actions

- 📞 Contact the customer within 24 hours.
- 🎁 Offer a personalized discount or loyalty reward.
- 📄 Recommend upgrading to a long-term contract.
- 💬 Conduct a customer satisfaction follow-up.
- 👨‍💼 Assign a customer success representative.

### Possible Reasons

- High monthly charges
- Month-to-month contract
- Lack of support services
- Fiber optic users generally show higher churn
- Electronic check payment method

### Business Impact

Retaining existing customers is significantly more cost-effective than acquiring new ones. Early identification of at-risk customers enables proactive retention strategies and helps reduce revenue loss.
""")

    else:

        st.subheader("🎉 Customer Retention Outlook")

        st.markdown("""
### Great News!

The customer is unlikely to churn.

### Recommended Actions

- 😊 Continue providing quality service.
- 🎁 Offer loyalty benefits.
- ⭐ Promote premium plans or additional services.
- 💬 Continue collecting customer feedback.

### Business Insight

Satisfied customers often generate repeat business and positive referrals, contributing to long-term business growth.
""")

    st.divider()

    # ==========================================
    # Prediction Summary
    # ==========================================

    summary = pd.DataFrame({
        "Feature": [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Tenure",
            "Internet Service",
            "Contract",
            "Monthly Charges",
            "Total Charges",
            "Prediction",
            "Probability"
        ],
        "Value": [
            gender,
            "Yes" if senior else "No",
            partner,
            dependents,
            tenure,
            internet_service,
            contract,
            monthly,
            total,
            "Churn" if prediction[0] == 1 else "Stay",
            f"{probability*100:.2f}%"
        ]
    })

    st.subheader("📋 Prediction Summary")

    st.dataframe(summary, use_container_width=True)

    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        file_name="customer_churn_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================
# Footer
# ==========================================

st.divider()

st.markdown(
    """
<div style='text-align:center;'>

### 📊 AI Customer Churn Prediction System

Developed using

🐍 Python |
📈 Scikit-learn |
🚀 Streamlit |
🤖 Logistic Regression

**End-to-End Machine Learning Project**

</div>
""",
unsafe_allow_html=True
)