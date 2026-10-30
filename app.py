
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model/churn_model.pkl")

# Load dataset for analytics
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Analytics Dashboard")
st.caption("Predict customer churn and understand customer retention patterns.")

# ---------------- ANALYTICS ----------------

st.header("Business Overview")

total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()
churn_rate = churned_customers / total_customers * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# ---------------- CHARTS ----------------

st.header("Customer Analytics")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Customer Churn Distribution")
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)

st.divider()


# ---------------- MODEL EXPLAINABILITY ----------------

st.divider()
st.header("Model Explainability")

st.write(
    "The following chart displays the strongest features "
    "associated with higher and lower churn predictions "
    "according to the Logistic Regression model."
)

# Extract model components
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get feature names and coefficients
feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Select strongest positive and negative coefficients
top_positive = importance_df.nlargest(8, "Coefficient")
top_negative = importance_df.nsmallest(8, "Coefficient")

explainability_df = pd.concat(
    [top_positive, top_negative]
).sort_values("Coefficient")

# Create chart
fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    explainability_df["Feature"],
    explainability_df["Coefficient"]
)

ax.axvline(0, linewidth=1)
ax.set_xlabel("Logistic Regression Coefficient")
ax.set_title("Features Associated with Customer Churn")

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Positive coefficients are associated with higher churn "
    "log-odds; negative coefficients are associated with lower "
    "churn log-odds. Coefficients are not causal effects."
)


# ---------------- CUSTOMER PREDICTION ----------------

st.header("Individual Customer Prediction")

st.write("Enter customer details to estimate their churn probability.")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.subheader("Billing Information")

charge_col1, charge_col2 = st.columns(2)

with charge_col1:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0
    )

with charge_col2:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0
    )

# ---------------- PREDICTION ----------------

st.divider()

if st.button("Predict Customer Churn", type="primary"):

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    churn_probability = model.predict_proba(customer_data)[0][1]

    threshold = 0.4
    prediction = int(churn_probability >= threshold)

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with result_col2:
        if churn_probability < 0.30:
            risk = "Low Risk"
        elif churn_probability <= 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        st.metric("Customer Risk Level", risk)

    st.progress(float(churn_probability))

    if prediction == 1:
        st.error("This customer is classified as likely to churn.")
    else:
        st.success("This customer is classified as unlikely to churn.")

    # Retention recommendations
    st.subheader("Suggested Retention Actions")

    if contract == "Month-to-month":
        st.write("- Consider offering an annual contract with suitable benefits.")

    if tenure <= 12:
        st.write("- Provide onboarding support and early-stage engagement.")

    if monthly_charges > 80:
        st.write("- Review the customer's plan and consider suitable pricing options.")

    if online_security == "No" and internet_service != "No":
        st.write("- Consider offering an online security package.")

    if tech_support == "No" and internet_service != "No":
        st.write("- Consider offering additional technical support.")

    if prediction == 1:
        st.info(
            "Prioritize this customer for a retention follow-up. "
            "These suggestions are rule-based, not guaranteed solutions."
        )



# ---------------- BATCH PREDICTION ----------------

st.divider()
st.header("Batch Customer Churn Prediction")

st.write(
    "Upload a CSV file containing multiple customer records "
    "to predict churn for all customers."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Customer Data")
    st.dataframe(batch_df.head())

    # Required model input columns
    required_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in batch_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )

    else:

        if st.button("Predict All Customers"):

            customer_features = batch_df[required_columns].copy()

            # Convert TotalCharges to numeric
            customer_features["TotalCharges"] = pd.to_numeric(
                customer_features["TotalCharges"],
                errors="coerce"
            )

            # Get probabilities
            probabilities = model.predict_proba(
                customer_features
            )[:, 1]

            # Apply threshold
            threshold = 0.4

            predictions = (probabilities >= threshold).astype(int)

            # Create results
            results = batch_df.copy()

            results["Churn Probability (%)"] = (
                probabilities * 100
            ).round(2)

            results["Churn Prediction"] = [
                "Likely to Churn" if p == 1
                else "Unlikely to Churn"
                for p in predictions
            ]

            results["Risk Level"] = [
                "Low Risk" if probability < 0.30
                else "Medium Risk" if probability <= 0.60
                else "High Risk"
                for probability in probabilities
            ]

            st.subheader("Batch Prediction Results")

            st.dataframe(results)

            # Summary metrics
            total = len(results)
            likely_to_churn = int(predictions.sum())
            unlikely_to_churn = total - likely_to_churn

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", total)
            col2.metric("Likely to Churn", likely_to_churn)
            col3.metric("Unlikely to Churn", unlikely_to_churn)

            # Download results
            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv"
            )

