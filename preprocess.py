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


RAW_DEFAULTS = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 0.0,
}


def _normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_choice(value, choices, default):
    normalized_value = _normalize_text(value).lower()
    lookup = {str(choice).lower(): choice for choice in choices}
    return lookup.get(normalized_value, default)


def _normalize_yes_no(value, default="No"):
    normalized_value = _normalize_text(value).lower()
    if normalized_value in {"yes", "y", "true", "1"}:
        return "Yes"
    if normalized_value in {"no", "n", "false", "0"}:
        return "No"
    return default


def _normalize_senior(value):
    normalized_value = _normalize_text(value).lower()
    if normalized_value in {"1", "yes", "y", "true"}:
        return 1
    return 0


def preprocess(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    for col, default_value in RAW_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    for col, default_value in RAW_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_value)

    df["gender"] = df["gender"].apply(lambda x: _normalize_choice(x, ["Male", "Female"], "Male"))
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(_normalize_senior)
    df["Partner"] = df["Partner"].apply(_normalize_yes_no)
    df["Dependents"] = df["Dependents"].apply(_normalize_yes_no)
    df["PhoneService"] = df["PhoneService"].apply(_normalize_yes_no, default="Yes")

    df["MultipleLines"] = df["MultipleLines"].apply(
        lambda x: _normalize_choice(x, ["Yes", "No", "No phone service"], "No")
    )

    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    df["InternetService"] = df["InternetService"].apply(
        lambda x: _normalize_choice(x, ["DSL", "Fiber optic", "No"], "DSL")
    )

    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: _normalize_choice(x, ["Yes", "No", "No internet service"], "No")
            )
            df[col] = df[col].replace({"No internet service": "No"})

    df["Contract"] = df["Contract"].apply(
        lambda x: _normalize_choice(x, ["Month-to-month", "One year", "Two year"], "Month-to-month")
    )
    df["PaperlessBilling"] = df["PaperlessBilling"].apply(_normalize_yes_no, default="Yes")
    df["PaymentMethod"] = df["PaymentMethod"].apply(
        lambda x: _normalize_choice(
            x,
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "Electronic check",
        )
    )

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["tenure"] = df["tenure"].fillna(df["tenure"].median() if not np.isnan(df["tenure"].median()) else 0)
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
        df["MonthlyCharges"].median() if not np.isnan(df["MonthlyCharges"].median()) else 70.0
    )

    fallback_total = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges"] = df["TotalCharges"].fillna(fallback_total)
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].mean() if not np.isnan(df["TotalCharges"].mean()) else 0.0
    )

    df["tenure"] = df["tenure"].clip(lower=0, upper=120)
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0, upper=400)
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)

    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], dtype=int)

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
