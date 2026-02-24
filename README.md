# Customer Churn Prediction

A machine learning-based web application that predicts customer churn in the telecom industry using Logistic Regression.

## Features

- **Single Customer Prediction** — Enter customer details manually and get instant churn probability
- **Batch Prediction** — Upload a CSV file to predict churn for multiple customers at once
- **Risk Classification** — Customers are categorized into Low, Medium, or High risk
- **Model Analysis** — View feature importance and model performance metrics

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy, Matplotlib

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Model Details

- **Algorithm:** Logistic Regression
- **Dataset:** Telco Customer Churn (Kaggle) — 7043 records
- **Features:** 24 input features after preprocessing
- **Accuracy:** 79.28%
