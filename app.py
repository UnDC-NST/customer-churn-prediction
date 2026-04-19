import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from preprocess import preprocess, columns
from retention_agent import AgenticRetentionAssistant

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-surface: #f4f8fc;
        --bg-card: rgba(255, 255, 255, 0.92);
        --ink-primary: #0f172a;
        --ink-secondary: #475467;
        --brand-primary: #006d77;
        --brand-secondary: #f4a261;
        --border-soft: #d9e3ec;
    }

    /* Main background */
    .stApp {
        background:
            radial-gradient(1200px 520px at 100% -5%, rgba(0, 109, 119, 0.18), transparent 55%),
            radial-gradient(900px 460px at -10% 10%, rgba(244, 162, 97, 0.15), transparent 60%),
            linear-gradient(180deg, #f8fbff 0%, #edf3f9 100%);
        color: var(--ink-primary);
    }

    .main .block-container {
        padding-top: 2.2rem;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--ink-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* General Text */
    .stMarkdown, p, li {
        color: var(--ink-secondary) !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border-soft);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(0,109,119,0.08), rgba(244,162,97,0.12));
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        animation: fadeUp 0.5s ease-out;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        right: -120px;
        top: -80px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,109,119,0.16) 0%, rgba(0,109,119,0.00) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.6rem, 3.2vw, 2.4rem);
        font-weight: 700;
        color: #0b1f3a;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 0.35rem;
        color: #344054;
        max-width: 940px;
    }

    .hero-badges {
        margin-top: 0.85rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .hero-badge {
        background: rgba(255,255,255,0.75);
        border: 1px solid #c8d8e3;
        color: #0b5660;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.22rem 0.72rem;
        font-size: 0.8rem;
        backdrop-filter: blur(2px);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border-soft);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--brand-primary);
        font-weight: 700;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #667085;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 1px solid #d4dee8;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px 10px 0 0;
        padding-top: 7px;
        padding-bottom: 7px;
        font-weight: 600;
        color: #667085;
        font-family: 'Manrope', sans-serif !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--brand-primary) !important;
        border-bottom: 2px solid var(--brand-primary) !important;
        background: rgba(0, 109, 119, 0.05);
    }

    /* Buttons */
    .stButton button, .stDownloadButton button {
        border-radius: 10px;
        border: 1px solid #005d66;
        background: linear-gradient(130deg, #006d77, #0a9396);
        color: #ffffff;
        font-weight: 700;
        font-family: 'Manrope', sans-serif;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(0, 109, 119, 0.22);
    }

    .stAlert {
        border-radius: 12px;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

with open("Model from Colab.pkl", "rb") as f:
    model = pickle.load(f)


def get_deployment_setting(name, default=""):
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
        if secret_value is None:
            return default
        secret_text = str(secret_value).strip()
        return secret_text if secret_text else default
    except Exception:
        return default


DEPLOYMENT_LLM_MODEL = get_deployment_setting("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
DEPLOYMENT_GROQ_KEY = get_deployment_setting("GROQ_API_KEY", "")
ASSISTANT_READY = bool(DEPLOYMENT_GROQ_KEY)


def risk_level(probability):
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


def render_retention_report(report):
    st.subheader("AI Retention Suggestions")
    st.markdown("Generated using model reasoning, knowledge retrieval, and a safety check.")

    st.markdown("### Churn Risk Summary")
    st.write(report.get("churn_risk_summary", "Not available."))

    segment = report.get("customer_segment", "Stable Core")
    strategy = report.get("segment_strategy", "Strategy unavailable.")
    st.markdown(f"**Customer Segment:** {segment}")
    st.markdown(f"**Segment Strategy:** {strategy}")

    st.markdown("### Key Contributing Drivers")
    for factor in report.get("key_contributing_factors", []):
        st.write(f"- {factor}")

    st.markdown("### Recommended Actions")
    actions = report.get("recommended_retention_actions", [])
    if actions:
        for idx, action in enumerate(actions, start=1):
            title = action.get("action", f"Action {idx}")
            rationale = action.get("rationale", "")
            priority = action.get("priority", "Medium")
            execution_notes = action.get("execution_notes", "")

            st.markdown(f"**{idx}. {title}**")
            st.write(f"Priority: {priority}")
            st.write(f"Rationale: {rationale}")
            if execution_notes:
                st.write(f"Execution notes: {execution_notes}")
    else:
        st.info("No actions were generated. In LLM-only mode, recommendations appear only when valid model output is available.")

    st.markdown("### Supporting Sources")
    for source in report.get("supporting_sources", []):
        source_line = f"- {source.get('id', '')}: [{source.get('title', 'Reference')}]({source.get('url', '')})"
        st.markdown(source_line)
        if source.get("justification"):
            st.write(f"  - Why relevant: {source.get('justification')}")

    st.markdown("### Notes")
    for disclaimer in report.get("business_and_ethical_disclaimers", []):
        st.write(f"- {disclaimer}")


enable_assistant = ASSISTANT_READY

assistant = None
if ASSISTANT_READY:
    assistant = AgenticRetentionAssistant(
        model=model,
        llm_model=DEPLOYMENT_LLM_MODEL,
        groq_api_key=DEPLOYMENT_GROQ_KEY,
    )

st.markdown(
    """
    <section class="hero-shell">
        <h1 class="hero-title">Customer Churn Prediction</h1>
        <p class="hero-subtitle">
            This app predicts churn risk for customers and gives AI-based retention suggestions.
            It supports both single input and batch CSV prediction.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Single + Batch Prediction</span>
            <span class="hero-badge">AI Suggestions</span>
            <span class="hero-badge">Model Insights</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Model Insights"])


with tab1:
    st.header("Project Goal")
    st.write(
        "Customer churn means users leaving a service. "
        "The goal of this project is to predict churn risk early so we can take action before customers leave."
    )

    st.header("What You Can Do")
    st.write(
        "You can test one customer manually or upload a CSV file for batch prediction. "
        "The app also gives simple AI suggestions for retention planning."
    )

    st.header("Input Data Format")
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

    st.header("Output Data Format")
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

    st.header("Processing Pipeline")
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
        "Select scoring mode:",
        ["Single Customer (Manual Input)", "Batch Prediction (CSV Upload)"],
        horizontal=True
    )

    if predict_mode == "Single Customer (Manual Input)":
        st.header("Single Customer Prediction")
        st.markdown("Enter customer details, run prediction, and view AI suggestions.")

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

            submitted = st.form_submit_button("Run Churn Score", use_container_width=True)

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

                risk = risk_level(prob_single)
                if risk == "Low":
                    risk_color = "#4caf50"
                elif risk == "Medium":
                    risk_color = "#ff9800"
                else:
                    risk_color = "#f44336"

                prediction_text = "Yes — Customer is likely to churn" if pred_single == 1 else "No — Customer is likely to stay"
                pred_color = "#f44336" if pred_single == 1 else "#4caf50"

                st.markdown("---")
                st.subheader("Scoring Result")

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric("Churn Probability", f"{prob_pct}%")
                with r_col2:
                    st.metric("Prediction", "Likely Churn" if pred_single == 1 else "Likely Retained")
                with r_col3:
                    st.metric("Risk Level", risk)

                st.markdown(f"""
                <div style="margin: 20px 0;">
                    <p style="color: #667085; font-size: 14px; margin-bottom: 8px;">Churn Probability Gauge</p>
                    <div style="background: #e8edf3; border-radius: 12px; height: 36px; width: 100%; position: relative; overflow: hidden; border: 1px solid #d3dde8;">
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
                            color: #ffffff;
                            font-weight: 700;
                            font-size: 16px;
                            text-shadow: 0 1px 3px rgba(0,0,0,0.7);
                        ">{prob_pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    background: #ffffff;
                    border-left: 5px solid {pred_color};
                    padding: 20px 24px;
                    border-radius: 8px;
                    margin: 16px 0;
                    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
                ">
                    <h4 style="margin: 0 0 8px 0; color: {pred_color};">{prediction_text}</h4>
                    <p style="margin: 0; color: #344054; font-size: 14px;">
                        This customer has a <strong style="color: {risk_color};">{risk} risk</strong> of churning
                        with a probability of <strong>{prob_pct}%</strong>.
                        {"Consider targeted retention offers such as discounts or contract upgrades." if pred_single == 1 else "This customer appears satisfied. Continue monitoring their engagement."}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Input Summary")
                summary_df = pd.DataFrame({
                    "Feature": list(single_data.keys()),
                    "Value": [str(v[0]) for v in single_data.values()]
                })
                st.dataframe(summary_df, width="stretch")

                if enable_assistant and assistant is not None:
                    st.markdown("---")
                    with st.spinner("Generating AI retention suggestions..."):
                        customer_profile = {key: value[0] for key, value in single_data.items()}
                        report = assistant.generate_retention_report(
                            customer_profile=customer_profile,
                            processed_features=processed_single.iloc[0].to_dict(),
                            churn_probability=float(prob_single),
                            churn_prediction=int(pred_single),
                        )
                    render_retention_report(report)
                else:
                    st.info("AI suggestions are unavailable. Add GROQ_API_KEY in deployment secrets.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.header("Batch Prediction")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.subheader("Raw Data Preview")
            st.dataframe(df_raw.head(10), width="stretch")
            st.write(f"Total records: {len(df_raw)}, Columns: {len(df_raw.columns)}")

            if st.button("Run Batch Scoring"):
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
                    result_df["Risk Level"] = [risk_level(p) for p in probs]

                    input_records = df_input.to_dict("records")
                    segment_summary = pd.DataFrame()

                    if enable_assistant and assistant is not None:
                        rag_segments = assistant.generate_batch_segment_strategies(
                            raw_df=df_input,
                            processed_df=processed,
                            churn_probabilities=probs,
                        )
                        row_segments = rag_segments.get("row_segments", [])
                        segment_summary = rag_segments.get("segment_summary", pd.DataFrame())

                        if len(row_segments) == len(result_df):
                            result_df["Customer Segment"] = row_segments
                        else:
                            result_df["Customer Segment"] = "Segment unavailable"

                        if not segment_summary.empty:
                            strategy_lookup = dict(
                                zip(segment_summary["Customer Segment"], segment_summary["Recommended Strategy"])
                            )
                            result_df["Segment Strategy"] = result_df["Customer Segment"].map(strategy_lookup)
                            result_df["Segment Strategy"] = result_df["Segment Strategy"].fillna("Strategy unavailable")
                        else:
                            result_df["Segment Strategy"] = "Strategy unavailable"
                    else:
                        result_df["Customer Segment"] = "AI unavailable"
                        result_df["Segment Strategy"] = "Add deployment key to generate strategy suggestions"

                    st.subheader("Batch Scoring Results")
                    st.dataframe(result_df, width="stretch")

                    col1, col2, col3 = st.columns(3)
                    total = len(preds)
                    churned = int(sum(preds))
                    col1.metric("Total Customers", total)
                    col2.metric("Predicted to Churn", churned)
                    col3.metric("Predicted Churn Rate", f"{churned / total * 100:.1f}%")

                    st.subheader("Risk Distribution")
                    risk_counts = result_df["Risk Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
                    
                    fig_risk, ax_risk = plt.subplots(figsize=(5, 3.5))
                    fig_risk.patch.set_facecolor('#f4f8fc')
                    ax_risk.set_facecolor('#f4f8fc')
                    bar_colors = ["#4caf50", "#ff9800", "#f44336"]
                    ax_risk.bar(risk_counts.index, risk_counts.values, color=bar_colors, edgecolor='#f4f8fc', linewidth=0.5)
                    ax_risk.set_ylabel("Number of Customers", fontsize=10, color="#334155")
                    ax_risk.set_title("Customers by Risk Level", fontsize=12, fontweight='bold', color="#0f172a", pad=10)
                    ax_risk.tick_params(colors='#334155')
                    ax_risk.spines['top'].set_visible(False)
                    ax_risk.spines['right'].set_visible(False)
                    ax_risk.spines['bottom'].set_color('#94a3b8')
                    ax_risk.spines['left'].set_color('#94a3b8')
                    plt.tight_layout()
                    
                    # Wrap in columns to limit width
                    col_chart, _ = st.columns([1, 1])
                    with col_chart:
                        st.pyplot(fig_risk)

                    st.subheader("Segment Strategies")
                    if enable_assistant and assistant is not None and not segment_summary.empty:
                        st.dataframe(segment_summary, width="stretch")

                        fig_seg, ax_seg = plt.subplots(figsize=(7, 3.8))
                        fig_seg.patch.set_facecolor('#f4f8fc')
                        ax_seg.set_facecolor('#f4f8fc')
                        ax_seg.barh(
                            segment_summary["Customer Segment"][::-1],
                            segment_summary["Customers"][::-1],
                            color="#4da8da",
                            edgecolor="#f4f8fc",
                            linewidth=0.5,
                        )
                        ax_seg.set_xlabel("Customers", fontsize=10, color="#334155")
                        ax_seg.set_title("Customer Cohorts", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
                        ax_seg.tick_params(colors="#334155")
                        ax_seg.spines['top'].set_visible(False)
                        ax_seg.spines['right'].set_visible(False)
                        ax_seg.spines['bottom'].set_color('#94a3b8')
                        ax_seg.spines['left'].set_color('#94a3b8')
                        plt.tight_layout()

                        col_seg_chart, _ = st.columns([2, 1])
                        with col_seg_chart:
                            st.pyplot(fig_seg)
                    else:
                        st.info("Enable AI suggestions to generate segment strategies.")

                    if enable_assistant and assistant is not None:
                        st.markdown("---")
                        st.subheader("Top At-Risk Customer: AI Retention Suggestions")
                        top_idx = int(np.argmax(probs))
                        top_profile = input_records[top_idx]
                        top_processed = processed.iloc[top_idx].to_dict()
                        top_prob = float(probs[top_idx])
                        top_pred = int(preds[top_idx])

                        with st.spinner("Generating report for the highest-risk customer..."):
                            top_report = assistant.generate_retention_report(
                                customer_profile=top_profile,
                                processed_features=top_processed,
                                churn_probability=top_prob,
                                churn_prediction=top_pred,
                            )
                        render_retention_report(top_report)
                    else:
                        st.info("AI suggestions are unavailable. Add GROQ_API_KEY in deployment secrets.")

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
                        fig_cm.patch.set_facecolor('#f4f8fc')
                        ax_cm.set_facecolor('#f4f8fc')
                        im = ax_cm.imshow(cm, cmap="Blues")
                        ax_cm.set_xticks([0, 1])
                        ax_cm.set_yticks([0, 1])
                        ax_cm.set_xticklabels(["No Churn", "Churn"], color="#334155")
                        ax_cm.set_yticklabels(["No Churn", "Churn"], color="#334155")
                        ax_cm.set_xlabel("Predicted", fontsize=9, color="#334155")
                        ax_cm.set_ylabel("Actual", fontsize=9, color="#334155")
                        ax_cm.set_title("Confusion Matrix", fontsize=11, fontweight='bold', color="#0f172a")
                        for i in range(2):
                            for j in range(2):
                                text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=11)
                        
                        cbar = plt.colorbar(im, ax=ax_cm)
                        cbar.ax.yaxis.set_tick_params(color='#334155')
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#334155')
                        plt.tight_layout()

                        fpr, tpr, _ = roc_curve(true_labels, probs)
                        roc_auc = auc(fpr, tpr)
                        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                        fig_roc.patch.set_facecolor('#f4f8fc')
                        ax_roc.set_facecolor('#f4f8fc')
                        ax_roc.plot(fpr, tpr, color="#4da8da", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
                        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
                        ax_roc.set_xlabel("False Positive Rate", fontsize=9, color="#334155")
                        ax_roc.set_ylabel("True Positive Rate", fontsize=9, color="#334155")
                        ax_roc.set_title("ROC Curve", fontsize=11, fontweight='bold', color="#0f172a")
                        ax_roc.tick_params(colors='#334155')
                        ax_roc.spines['top'].set_visible(False)
                        ax_roc.spines['right'].set_visible(False)
                        ax_roc.spines['bottom'].set_color('#94a3b8')
                        ax_roc.spines['left'].set_color('#94a3b8')
                        
                        legend = ax_roc.legend(fontsize=8, loc="lower right")
                        plt.setp(legend.get_texts(), color='#334155')
                        legend.get_frame().set_facecolor('#ffffff')
                        legend.get_frame().set_edgecolor('#d9e3ec')
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
    st.header("Feature Impact Analysis")
    st.write(
        "Feature impact is derived from absolute logistic regression coefficients. "
        "Higher magnitude indicates stronger directional influence on churn probability."
    )

    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": columns,
        "Coefficient": coefficients,
        "Absolute Importance": np.abs(coefficients)
    }).sort_values("Absolute Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
    fig_imp.patch.set_facecolor('#f4f8fc')
    ax_imp.set_facecolor('#f4f8fc')
    top10 = importance_df.head(10)
    colors_imp = ["#e53935" if c > 0 else "#4da8da" for c in top10["Coefficient"]]
    ax_imp.barh(top10["Feature"][::-1], top10["Absolute Importance"][::-1], color=colors_imp[::-1], edgecolor='#f4f8fc', linewidth=0.5)
    ax_imp.set_xlabel("Absolute Coefficient Value", fontsize=10, color="#334155")
    ax_imp.set_title("Top 10 Churn Drivers", fontsize=12, fontweight='bold', color="#0f172a")
    ax_imp.tick_params(colors='#334155')
    ax_imp.spines['top'].set_visible(False)
    ax_imp.spines['right'].set_visible(False)
    ax_imp.spines['bottom'].set_color('#94a3b8')
    ax_imp.spines['left'].set_color('#94a3b8')
    plt.tight_layout()
    
    col_chart2, _ = st.columns([2, 1])
    with col_chart2:
        st.pyplot(fig_imp)

    st.write(
        "Red bars indicate features that positively drive churn risk. "
        "Blue bars indicate features that negatively drive churn risk (retention factors)."
    )
    st.dataframe(importance_df.reset_index(drop=True), width="stretch")

    st.header("Model Metadata")
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

    st.header("Performance Summary")
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

    st.header("Model Considerations")
    limitations = [
        "The model uses a fixed threshold of 0.5 for binary classification, which may need tuning for business needs.",
        "Logistic regression assumes linear decision boundaries, which may underfit complex churn patterns.",
        "The dataset is imbalanced at approximately 27% churn, which reduces recall on the minority class.",
        "TotalCharges missing values are imputed with the column mean, which may introduce bias for new customers.",
        "The model does not capture temporal trends or sequential customer behavior over time."
    ]
    for lim in limitations:
        st.write(f"- {lim}")
