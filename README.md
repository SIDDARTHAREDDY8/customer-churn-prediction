# 🧠 Customer Churn Prediction App

An interactive machine learning web app built with **Streamlit** to predict customer churn for a telecom company.  
It supports both **single customer prediction** and **batch predictions** via CSV upload.

---

## 🔗 Live Demo

👉 [Click here to try the app](https://customer-churn-prediction-5nkl8w9bf6cxvgguravfav.streamlit.app)

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?logo=streamlit)](https://customer-churn-prediction-5nkl8w9bf6cxvgguravfav.streamlit.app)

---

## 🧩 Problem Statement

Customer churn is a critical issue in the telecom industry. Retaining existing customers is more cost-effective than acquiring new ones.  
This app predicts whether a customer is likely to churn based on their profile and service usage.
---
## 🚀 Features

### 🔍 1. Single Prediction
- Enter details of a single customer.
- Get real-time churn prediction with probability score.
- Simple and clean form interface.

### 📂 2. Batch Prediction
- Upload a CSV file of customer records.
- Get churn predictions for the entire dataset.
- Download results as a CSV with added predictions and probabilities.

### 📊 3. Insights & Visualizations
- Summary KPIs: Churn Rate, Monthly Charges, Tenure, Customer Count.
- Interactive charts using **Plotly**:
  - 📂 Contract Type vs Churn
  - 💸 Monthly Charges vs Churn
  - ⏳ Tenure Distribution by Churn
  - 🌐 Internet Service vs Churn
  - 🎬 StreamingTV & TechSupport vs Churn
  - 🔥 Correlation Heatmap
  - 🥧 Churn Pie Chart

---
## 🧰 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Jupyter Notebooks**
- **Matplotlib / Seaborn (for EDA)**

---

## 📸 App Screenshot

Here’s what the app looks like:

![App Screenshot](https://github.com/SIDDARTHAREDDY8/customer-churn-prediction/blob/main/images/Screenshot%202025-08-06%20at%205.40.57%E2%80%AFPM.png?raw=true)
![App Screenshot](https://github.com/SIDDARTHAREDDY8/customer-churn-prediction/blob/main/images/Screenshot%202025-08-06%20at%205.46.40%E2%80%AFPM.png?raw=true)
![App Screenshot](https://github.com/SIDDARTHAREDDY8/customer-churn-prediction/blob/main/images/Screenshot%202025-08-06%20at%205.46.53%E2%80%AFPM.png?raw=true)
![App Screenshot](https://github.com/SIDDARTHAREDDY8/customer-churn-prediction/blob/main/images/Screenshot%202025-08-06%20at%205.47.46%E2%80%AFPM.png?raw=true)
![App Screenshot](https://github.com/SIDDARTHAREDDY8/customer-churn-prediction/blob/main/images/Screenshot%202025-08-06%20at%205.47.53%E2%80%AFPM.png?raw=true)

---
## 📁 Folder Structure

customer-churn-prediction/
├── app/
│ ├── app.py # Streamlit app
│ └── model.pkl # Trained ML model
├── data/
│ └── customer_churn.csv # Sample dataset
├── notebooks/
│ └── train_model.py # Model training script
├── images/
│ └── preview.png # App preview screenshot (optional)
├── requirements.txt # Python dependencies
├── README.md # Project readme


---

## 🏁 How to Run Locally

1. **Clone the repo:**

```bash
git clone https://github.com/SIDDARTHAREDDY8/customer-churn-prediction.git
cd customer-churn-prediction
```
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```
3. **Run the app:**

```bash
cd app
streamlit run app.py
```
---
## 👤 Author

**[Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)**  
🎓 Master’s in CS | Aspiring Data Scientist  
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/siddarthareddy9)

⭐️ Show Some Love
If you like this project, don’t forget to ⭐️ the repo and share it!
