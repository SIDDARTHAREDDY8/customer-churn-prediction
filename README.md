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

- 🔍 **Single Prediction** – Predict churn for an individual customer via form inputs.
- 📂 **Batch Prediction** – Upload a CSV file with customer details for bulk predictions.
- 📊 **Interactive UI** – Built with Streamlit for a smooth user experience.
- 🧠 **Machine Learning Model** – Trained using `scikit-learn` and saved using `joblib`.

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
