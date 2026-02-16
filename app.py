import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from preprocess import preprocess, columns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

with open("Model from Colab.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction System")
st.markdown("Milestone 1 — ML-Based Customer Churn Prediction")

tab1, tab2, tab3 = st.tabs(["Project Overview", "Predict", "Model Analysis"])


with tab1:
    st.header("Problem Statement")
    st.write(
        "Customer churn occurs when a customer stops doing business with a company. "
        "In the telecom industry, the annual churn rate ranges between 15 and 25 percent due to intense competition. "
        "Retaining an existing customer costs significantly less than acquiring a new one. "
        "This system identifies customers at high risk of churning so that targeted retention actions can be taken."
    )

    st.header("Business Use Case")
    st.write(
        "A telecom provider wants to proactively reach out to customers who are likely to cancel their subscription. "
        "By predicting churn before it happens, the business can offer personalized incentives and reduce revenue loss."
    )

    st.header("Input Specification")
    input_data = {
        "Column": [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ],
        "Type": [
            "string", "categorical", "binary int", "categorical", "categorical",
            "integer", "categorical", "categorical", "categorical",
            "categorical", "categorical", "categorical", "categorical",
            "categorical", "categorical", "categorical", "categorical",
            "categorical", "float", "string/float"
        ],
        "Description": [
            "Unique customer identifier (dropped before model input)",
            "Male or Female",
            "1 if senior citizen, 0 otherwise",
            "Yes or No — whether the customer has a partner",
            "Yes or No — whether the customer has dependents",
            "Number of months the customer has been with the company",
            "Yes or No",
            "Yes, No, or No phone service",
            "DSL, Fiber optic, or No",
            "Yes, No, or No internet service",
            "Yes, No, or No internet service",
            "Yes, No, or No internet service",
            "Yes, No, or No internet service",
            "Yes, No, or No internet service",
            "Yes, No, or No internet service",
            "Month-to-month, One year, or Two year",
            "Yes or No",
            "Electronic check, Mailed check, Bank transfer, Credit card",
            "Monthly bill amount in USD",
            "Total amount charged (may be blank for new customers)"
        ]
    }
    st.dataframe(pd.DataFrame(input_data), width="stretch")

    st.header("Output Specification")
    output_data = {
        "Output": ["Churn Probability", "Prediction", "Risk Level"],
        "Type": ["float 0-100%", "Yes / No", "Low / Medium / High"],
        "Description": [
            "Probability that the customer will churn",
            "Binary classification result",
            "Low below 30%, Medium 30-60%, High above 60%"
        ]
    }
    st.dataframe(pd.DataFrame(output_data), width="stretch")

    st.header("System Architecture")
    st.code("""
Customer CSV
    |
    v
[Data Ingestion]
    |
    v
[Preprocessing Pipeline]
    - Drop customerID
    - Replace "No phone service" -> "No"
    - Replace "No internet service" -> "No"
    - One-hot encode: InternetService, PaymentMethod
    - Label encode: binary and ordinal columns
    - Impute missing TotalCharges with column mean
    - Align columns to training schema
    - Standard scale: tenure, MonthlyCharges, TotalCharges
    |
    v
[Logistic Regression Model]
    - Trained on Telco Customer Churn dataset (7043 records)
    - 24 input features
    - Binary output: Churn = Yes / No
    |
    v
[Output Layer]
    - Churn probability
    - Binary prediction
    - Risk level categorization
    - Downloadable results CSV
""", language="text")


with tab2:
    st.header("Upload Customer Data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(10), width="stretch")
        st.write(f"Total records: {len(df_raw)}, Columns: {len(df_raw.columns)}")

        if st.button("Predict Churn"):
            try:
                has_labels = "Churn" in df_raw.columns
                df_input = df_raw.copy()

                if has_labels:
                    true_labels = df_input["Churn"].map({"Yes": 1, "No": 0}).values
                    df_input = df_input.drop("Churn", axis=1)
                else:
                    true_labels = None

                processed = preprocess(df_input.copy())

                probs = model.predict_proba(processed)[:, 1]
                preds = model.predict(processed)

                result_df = df_raw.copy()
                result_df["Churn Probability"] = (probs * 100).round(2).astype(str) + "%"
                result_df["Prediction"] = ["Yes" if p == 1 else "No" for p in preds]

                def risk_level(x):
                    if x < 0.3:
                        return "Low"
                    elif x < 0.6:
                        return "Medium"
                    else:
                        return "High"

                result_df["Risk Level"] = [risk_level(p) for p in probs]

                st.subheader("Prediction Results")
                st.dataframe(result_df, width="stretch")

                col1, col2, col3 = st.columns(3)
                total = len(preds)
                churned = int(sum(preds))
                col1.metric("Total Customers", total)
                col2.metric("Predicted to Churn", churned)
                col3.metric("Predicted Churn Rate", f"{churned / total * 100:.1f}%")

                st.subheader("Risk Distribution")
                risk_counts = result_df["Risk Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
                fig_risk, ax_risk = plt.subplots(figsize=(5, 3))
                bar_colors = ["#4caf50", "#ff9800", "#f44336"]
                ax_risk.bar(risk_counts.index, risk_counts.values, color=bar_colors)
                ax_risk.set_ylabel("Number of Customers")
                ax_risk.set_title("Customers by Risk Level")
                plt.tight_layout()
                st.pyplot(fig_risk)

                if has_labels and true_labels is not None:
                    st.subheader("Model Evaluation on Uploaded Data")
                    acc = accuracy_score(true_labels, preds)
                    prec = precision_score(true_labels, preds)
                    rec = recall_score(true_labels, preds)
                    f1 = f1_score(true_labels, preds)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.4f}")
                    m2.metric("Precision", f"{prec:.4f}")
                    m3.metric("Recall", f"{rec:.4f}")
                    m4.metric("F1 Score", f"{f1:.4f}")

                    cm = confusion_matrix(true_labels, preds)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    im = ax_cm.imshow(cm, cmap="Blues")
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(["No Churn", "Churn"])
                    ax_cm.set_yticklabels(["No Churn", "Churn"])
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.set_title("Confusion Matrix")
                    for i in range(2):
                        for j in range(2):
                            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
                    plt.colorbar(im, ax=ax_cm)
                    plt.tight_layout()
                    st.pyplot(fig_cm)

                    fpr, tpr, _ = roc_curve(true_labels, probs)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    ax_roc.plot(fpr, tpr, color="steelblue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
                    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("ROC Curve")
                    ax_roc.legend()
                    plt.tight_layout()
                    st.pyplot(fig_roc)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload a customer CSV file to begin.")


with tab3:
    st.header("Feature Importance")
    st.write(
        "Feature importance is derived from the absolute values of the logistic regression coefficients. "
        "A higher magnitude indicates stronger influence on the churn prediction output."
    )

    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": columns,
        "Coefficient": coefficients,
        "Absolute Importance": np.abs(coefficients)
    }).sort_values("Absolute Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    top10 = importance_df.head(10)
    colors_imp = ["#d32f2f" if c > 0 else "#1565c0" for c in top10["Coefficient"]]
    ax_imp.barh(top10["Feature"][::-1], top10["Absolute Importance"][::-1], color=colors_imp[::-1])
    ax_imp.set_xlabel("Absolute Coefficient Value")
    ax_imp.set_title("Top 10 Churn-Driving Features")
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.write(
        "Red bars indicate features that positively drive churn risk. "
        "Blue bars indicate features that negatively drive churn risk (retention factors)."
    )
    st.dataframe(importance_df.reset_index(drop=True), width="stretch")

    st.header("Model Information")
    info = {
        "Attribute": [
            "Algorithm", "Training Dataset", "Training Records",
            "Number of Features", "Target Variable",
            "Positive Class", "Scaling"
        ],
        "Details": [
            "Logistic Regression (max_iter=2000)",
            "Telco Customer Churn — Kaggle",
            "7043",
            "24",
            "Churn",
            "Yes (customer churns)",
            "StandardScaler on tenure, MonthlyCharges, TotalCharges"
        ]
    }
    st.dataframe(pd.DataFrame(info), width="stretch")

    st.header("Model Performance Summary")
    st.write(
        "The model was trained on the full dataset using an 80/20 train-test split with random_state=32. "
        "Logistic Regression was selected as the final model after comparing KNN, Decision Tree, and Random Forest. "
        "It achieved the best balance between accuracy and recall on the held-out test set."
    )
    perf_data = {
        "Metric": ["Accuracy", "Precision (Churn=Yes)", "Recall (Churn=Yes)", "F1 Score (Churn=Yes)"],
        "Value": ["79.28%", "70%", "51%", "59%"]
    }
    st.dataframe(pd.DataFrame(perf_data), width="stretch")

    st.header("Limitations of Traditional ML Approach")
    limitations = [
        "The model uses a fixed threshold of 0.5 for binary classification, which may need tuning for business needs.",
        "Logistic regression assumes linear decision boundaries, which may underfit complex churn patterns.",
        "The dataset is imbalanced at approximately 27% churn, which reduces recall on the minority class.",
        "TotalCharges missing values are imputed with the column mean, which may introduce bias for new customers.",
        "The model does not capture temporal trends or sequential customer behavior over time."
    ]
    for lim in limitations:
        st.write(f"- {lim}")
