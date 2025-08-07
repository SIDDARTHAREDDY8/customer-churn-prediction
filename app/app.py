import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # <-- This is the missing line


# Load model
model = joblib.load("app/model.pkl")

# Set up page
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction App")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Insights"])

# Label mapping
label_map = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "No internet service": 2, "No phone service": 2,
    "DSL": 0, "Fiber optic": 1, "No": 2,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}

# ----------------------------- TAB 1 -----------------------------
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
        label_map[gender], senior, label_map[partner], label_map[dependents], tenure,
        label_map[phone_service], label_map[multiple_lines], label_map[internet_service],
        label_map[online_security], label_map[online_backup], label_map[device_protection],
        label_map[tech_support], label_map[streaming_tv], label_map[streaming_movies],
        label_map[contract], label_map[paperless_billing], label_map[payment_method],
        monthly_charges, total_charges
    ]])

    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn.\n\n💡 Probability: {prob:.2f}%")
        else:
            st.success(f"✅ The customer is not likely to churn.\n\n💡 Probability: {prob:.2f}%")


# ----------------------------- TAB 2 -----------------------------
with tab2:
    st.subheader("Upload a CSV of customer data")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Preview of uploaded data:")
            st.dataframe(df.head())

            df_processed = df.copy()
            for column in df_processed.columns:
                if df_processed[column].dtype == 'object':
                    df_processed[column] = df_processed[column].astype(str).map(label_map)

            churn_preds = model.predict(df_processed)
            churn_probs = model.predict_proba(df_processed)[:, 1]

            df['Churn_Prediction'] = np.where(churn_preds == 1, "Yes", "No")
            df['Churn_Probability (%)'] = (churn_probs * 100).round(2)

            st.success("✅ Predictions generated!")
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# ---------------------------------------------
# ✅ TAB 3: INSIGHTS & VISUALIZATIONS
# ---------------------------------------------
with tab3:
    st.subheader("📊 Visual Insights from Sample Data")

    try:
        df_sample = pd.read_csv("app/sample_data.csv")

        # -------------------------------
        # 📌 KPI Summary Cards
        # -------------------------------
        total_customers = len(df_sample)
        churn_rate = df_sample['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
        avg_monthly_charges = df_sample['MonthlyCharges'].mean()
        avg_tenure = df_sample['tenure'].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔁 Total Customers", total_customers)
        col2.metric("⚠️ Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("💰 Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
        col4.metric("📦 Avg Tenure (Months)", f"{avg_tenure:.1f}")

        st.markdown("---")

        # -------------------------------
        # 📂 Contract Type Distribution
        # -------------------------------
        st.markdown("### 📂 Contract Type Distribution")
        fig1 = px.histogram(df_sample, x="Contract", color="Churn", barmode="group", title="Contract Type by Churn")
        st.plotly_chart(fig1, use_container_width=True)

        # -------------------------------
        # 💸 Monthly Charges vs Churn
        # -------------------------------
        st.markdown("### 💸 Monthly Charges vs Churn")
        fig2 = px.histogram(df_sample, x="MonthlyCharges", color="Churn", nbins=50, barmode="overlay", opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------
        # ⏳ Tenure Distribution by Churn
        # -------------------------------
        st.markdown("### ⏳ Tenure Distribution by Churn")
        fig3 = px.histogram(df_sample, x="tenure", color="Churn", nbins=50, barmode="stack")
        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------------
        # 🌐 Churn by Internet Service
        # -------------------------------
        st.markdown("### 🌐 Churn by Internet Service")
        fig4 = px.bar(df_sample, x="InternetService", color="Churn", barmode="group")
        st.plotly_chart(fig4, use_container_width=True)

        # -------------------------------
        # 🎬 Churn Rate by StreamingTV & TechSupport
        # -------------------------------
        st.markdown("### 🎬 Churn by StreamingTV & Tech Support")

        col5, col6 = st.columns(2)
        with col5:
            fig5 = px.histogram(df_sample, x="StreamingTV", color="Churn", barmode="group")
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            fig6 = px.histogram(df_sample, x="TechSupport", color="Churn", barmode="group")
            st.plotly_chart(fig6, use_container_width=True)

        # -------------------------------
        # 🔥 Correlation Heatmap
        # -------------------------------
        st.markdown("### 🔥 Feature Correlation Heatmap")
        corr_df = df_sample.copy()
        corr_df = corr_df.drop(columns=['customerID'], errors='ignore')
        corr_df = corr_df.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        corr_df = pd.get_dummies(corr_df)

        corr = corr_df.corr()
        fig7 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig7, use_container_width=True)

        # -------------------------------
        # 🥧 Churn Pie Chart
        # -------------------------------
        st.markdown("### 🥧 Churn Distribution")
        churn_counts = df_sample['Churn'].value_counts()
        fig8 = px.pie(names=churn_counts.index, values=churn_counts.values, title="Churn vs Not Churned")
        st.plotly_chart(fig8, use_container_width=True)

    except Exception as e:
        st.error(f"Please ensure `app/sample_data.csv` is available.\n\n**Error loading data:** {e}")
