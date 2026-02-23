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

# Custom CSS for dark styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }

    /* General Text */
    .stMarkdown, p, li {
        color: #e0e0e0 !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #333333;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4da8da;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #b0bec5;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 2px solid #333333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #b0bec5;
    }
    .stTabs [aria-selected="true"] {
        color: #4da8da !important;
        border-bottom: 2px solid #4da8da !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #333333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

with open("Model from Colab.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction System")
st.markdown("Milestone 1 - ML-Based Customer Churn Prediction")

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
    predict_mode = st.radio(
        "Choose prediction mode:",
        ["Single Customer (Manual Input)", "Batch Prediction (CSV Upload)"],
        horizontal=True
    )

    if predict_mode == "Single Customer (Manual Input)":
        st.header("Single Customer Prediction")
        st.markdown("Fill in the customer details below and click **Predict** to see the churn risk.")

        with st.form("single_customer_form"):
            st.subheader("Personal Information")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
            with p_col2:
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            with p_col3:
                partner = st.selectbox("Partner", ["Yes", "No"])

            p_col4, p_col5 = st.columns(2)
            with p_col4:
                dependents = st.selectbox("Dependents", ["Yes", "No"])
            with p_col5:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

            st.subheader("Services")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            with s_col2:
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            with s_col3:
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            s_col4, s_col5, s_col6 = st.columns(3)
            with s_col4:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            with s_col5:
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            with s_col6:
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

            s_col7, s_col8, s_col9 = st.columns(3)
            with s_col7:
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            with s_col8:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            with s_col9:
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            st.subheader("Billing")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            with b_col2:
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

            b_col3, b_col4, b_col5 = st.columns(3)
            with b_col3:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
                )
            with b_col4:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            with b_col5:
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

            submitted = st.form_submit_button("Predict Churn", use_container_width=True)

        if submitted:
            try:
                senior_val = 1 if senior_citizen == "Yes" else 0

                single_data = {
                    "gender": [gender],
                    "SeniorCitizen": [senior_val],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "InternetService": [internet_service],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless_billing],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [str(total_charges)],
                }

                df_single = pd.DataFrame(single_data)
                processed_single = preprocess(df_single.copy())

                prob_single = model.predict_proba(processed_single)[:, 1][0]
                pred_single = model.predict(processed_single)[0]
                prob_pct = round(prob_single * 100, 2)

                if prob_single < 0.3:
                    risk = "Low"
                    risk_color = "#4caf50"
                    risk_emoji = ""
                elif prob_single < 0.6:
                    risk = "Medium"
                    risk_color = "#ff9800"
                    risk_emoji = ""
                else:
                    risk = "High"
                    risk_color = "#f44336"
                    risk_emoji = ""

                prediction_text = "Yes — Customer is likely to churn" if pred_single == 1 else "No — Customer is likely to stay"
                pred_color = "#f44336" if pred_single == 1 else "#4caf50"

                st.markdown("---")
                st.subheader("Prediction Result")

                # Result cards
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric("Churn Probability", f"{prob_pct}%")
                with r_col2:
                    st.metric("Prediction", "Churn" if pred_single == 1 else "No Churn")
                with r_col3:
                    st.metric("Risk Level", risk)

                # Visual probability gauge
                st.markdown(f"""
                <div style="margin: 20px 0;">
                    <p style="color: #b0bec5; font-size: 14px; margin-bottom: 8px;">Churn Probability Gauge</p>
                    <div style="background: #1e1e2e; border-radius: 12px; height: 36px; width: 100%; position: relative; overflow: hidden; border: 1px solid #333;">
                        <div style="
                            background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);
                            height: 100%;
                            width: {prob_pct}%;
                            border-radius: 12px 0 0 12px;
                            transition: width 0.5s ease;
                        "></div>
                        <span style="
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            color: white;
                            font-weight: 700;
                            font-size: 16px;
                            text-shadow: 0 1px 3px rgba(0,0,0,0.7);
                        ">{prob_pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Detailed result box
                st.markdown(f"""
                <div style="
                    background: #1a1a2e;
                    border-left: 5px solid {pred_color};
                    padding: 20px 24px;
                    border-radius: 8px;
                    margin: 16px 0;
                ">
                    <h4 style="margin: 0 0 8px 0; color: {pred_color};">{prediction_text}</h4>
                    <p style="margin: 0; color: #ccc; font-size: 14px;">
                        This customer has a <strong style="color: {risk_color};">{risk} risk</strong> of churning
                        with a probability of <strong>{prob_pct}%</strong>.
                        {"Consider targeted retention offers such as discounts or contract upgrades." if pred_single == 1 else "This customer appears satisfied. Continue monitoring their engagement."}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Input summary table
                st.subheader("Input Summary")
                summary_df = pd.DataFrame({
                    "Feature": list(single_data.keys()),
                    "Value": [str(v[0]) for v in single_data.values()]
                })
                st.dataframe(summary_df, width="stretch")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        # ---- Batch CSV Upload ----
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
                    
                    # Make the graph smaller and more styled
                    fig_risk, ax_risk = plt.subplots(figsize=(5, 3.5))
                    fig_risk.patch.set_facecolor('#0e1117')
                    ax_risk.set_facecolor('#0e1117')
                    bar_colors = ["#4caf50", "#ff9800", "#f44336"]
                    ax_risk.bar(risk_counts.index, risk_counts.values, color=bar_colors, edgecolor='#ffffff', linewidth=0.5)
                    ax_risk.set_ylabel("Number of Customers", fontsize=10, color="white")
                    ax_risk.set_title("Customers by Risk Level", fontsize=12, fontweight='bold', color="white", pad=10)
                    ax_risk.tick_params(colors='white')
                    ax_risk.spines['top'].set_visible(False)
                    ax_risk.spines['right'].set_visible(False)
                    ax_risk.spines['bottom'].set_color('white')
                    ax_risk.spines['left'].set_color('white')
                    plt.tight_layout()
                    
                    # Wrap in columns to limit width
                    col_chart, _ = st.columns([1, 1])
                    with col_chart:
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
                        fig_cm.patch.set_facecolor('#0e1117')
                        ax_cm.set_facecolor('#0e1117')
                        im = ax_cm.imshow(cm, cmap="Blues")
                        ax_cm.set_xticks([0, 1])
                        ax_cm.set_yticks([0, 1])
                        ax_cm.set_xticklabels(["No Churn", "Churn"], color="white")
                        ax_cm.set_yticklabels(["No Churn", "Churn"], color="white")
                        ax_cm.set_xlabel("Predicted", fontsize=9, color="white")
                        ax_cm.set_ylabel("Actual", fontsize=9, color="white")
                        ax_cm.set_title("Confusion Matrix", fontsize=11, fontweight='bold', color="white")
                        for i in range(2):
                            for j in range(2):
                                text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=11)
                        
                        cbar = plt.colorbar(im, ax=ax_cm)
                        cbar.ax.yaxis.set_tick_params(color='white')
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                        plt.tight_layout()

                        fpr, tpr, _ = roc_curve(true_labels, probs)
                        roc_auc = auc(fpr, tpr)
                        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                        fig_roc.patch.set_facecolor('#0e1117')
                        ax_roc.set_facecolor('#0e1117')
                        ax_roc.plot(fpr, tpr, color="#4da8da", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
                        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
                        ax_roc.set_xlabel("False Positive Rate", fontsize=9, color="white")
                        ax_roc.set_ylabel("True Positive Rate", fontsize=9, color="white")
                        ax_roc.set_title("ROC Curve", fontsize=11, fontweight='bold', color="white")
                        ax_roc.tick_params(colors='white')
                        ax_roc.spines['top'].set_visible(False)
                        ax_roc.spines['right'].set_visible(False)
                        ax_roc.spines['bottom'].set_color('white')
                        ax_roc.spines['left'].set_color('white')
                        
                        legend = ax_roc.legend(fontsize=8, loc="lower right")
                        plt.setp(legend.get_texts(), color='white')
                        legend.get_frame().set_facecolor('#0e1117')
                        legend.get_frame().set_edgecolor('white')
                        plt.tight_layout()

                        # Put charts side-by-side to control width and layout
                        col_cm, col_roc = st.columns(2)
                        with col_cm:
                            st.pyplot(fig_cm)
                        with col_roc:
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

    fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
    fig_imp.patch.set_facecolor('#0e1117')
    ax_imp.set_facecolor('#0e1117')
    top10 = importance_df.head(10)
    colors_imp = ["#e53935" if c > 0 else "#4da8da" for c in top10["Coefficient"]]
    ax_imp.barh(top10["Feature"][::-1], top10["Absolute Importance"][::-1], color=colors_imp[::-1], edgecolor='#ffffff', linewidth=0.5)
    ax_imp.set_xlabel("Absolute Coefficient Value", fontsize=10, color="white")
    ax_imp.set_title("Top 10 Churn-Driving Features", fontsize=12, fontweight='bold', color="white")
    ax_imp.tick_params(colors='white')
    ax_imp.spines['top'].set_visible(False)
    ax_imp.spines['right'].set_visible(False)
    ax_imp.spines['bottom'].set_color('white')
    ax_imp.spines['left'].set_color('white')
    plt.tight_layout()
    
    # Wrap in columns to limit width
    col_chart2, _ = st.columns([2, 1])
    with col_chart2:
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
