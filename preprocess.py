import pandas as pd
import pickle
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

with open("Columns (1).pkl", "rb") as f:
    columns = pickle.load(f)

with open("Scaler Model.pkl", "rb") as f:
    scaler = pickle.load(f)


def preprocess(df):
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], dtype=int)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling"
    ]

    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df
