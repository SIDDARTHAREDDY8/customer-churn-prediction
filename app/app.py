import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("app/model.pkl")
  # This is correct if app.py and model.pkl are in the same folder


# App layout
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction")

# Tabs for single vs batch
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# Label map (used for both tabs)
label_map = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "No internet service": 2, "No phone service": 2,
    "DSL": 0, "Fiber optic": 1, "No": 2,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}

# ---------------------------------------------
# ✅ TAB 1: SINGLE CUSTOMER PREDICTION
# ---------------------------------------------
with tab1:
    st.subheader("Enter a single customer's details:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

    input_data = np.array([[
        label_map[gender],
        senior,
        label_map[partner],
        label_map[dependents],
        tenure,
        label_map[phone_service],
        label_map[multiple_lines],
        label_map[internet_service],
        label_map[online_security],
        label_map[online_backup],
        label_map[device_protection],
        label_map[tech_support],
        label_map[streaming_tv],
        label_map[streaming_movies],
        label_map[contract],
        label_map[paperless_billing],
        label_map[payment_method],
        monthly_charges,
        total_charges
    ]])

    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn.\n\n💡 Probability: {prob:.2f}%")
        else:
            st.success(f"✅ The customer is not likely to churn.\n\n💡 Probability: {prob:.2f}%")


# ---------------------------------------------
# ✅ TAB 2: BATCH CUSTOMER PREDICTION
# ---------------------------------------------
with tab2:
    st.subheader("Upload a CSV of customer data")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Preview of uploaded data:")
            st.dataframe(df.head())

            # Preprocess the batch input (like during training)
            df_processed = df.copy()
            for column in df_processed.columns:
                if df_processed[column].dtype == 'object':
                    df_processed[column] = df_processed[column].astype(str).map(label_map)

            # Predict
            churn_preds = model.predict(df_processed)
            churn_probs = model.predict_proba(df_processed)[:, 1]

            df['Churn_Prediction'] = np.where(churn_preds == 1, "Yes", "No")
            df['Churn_Probability (%)'] = (churn_probs * 100).round(2)

            st.success("✅ Predictions generated!")
            st.dataframe(df)

            # Download predictions
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
