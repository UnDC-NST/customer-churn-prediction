# Customer Churn Prediction

This is a college project with a clean UI and practical workflow.
It predicts churn using a trained machine learning model and gives AI-based retention suggestions.

## Features

- Single customer churn prediction
- Batch prediction using CSV upload
- Churn probability, class label, and risk level
- AI suggestions for retention actions
- Model insight charts and evaluation metrics

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas, numpy, matplotlib
- LangGraph + LangChain + Groq
- python-dotenv

## Project Files

- app.py: main Streamlit app
- preprocess.py: data cleaning and feature prep
- retention_agent.py: AI workflow for suggestions
- retention_knowledge_base.py: reference data for retrieval
- requirements.txt: Python packages

## Run Locally

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Set environment values.

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
```

4. Start the app.

```bash
streamlit run app.py
```

## Optional Deployment Notes

- Keep API keys in server-side secrets.
- Do not commit .env files.
- In Streamlit Cloud, add GROQ_API_KEY and GROQ_MODEL_NAME in app secrets.

## Model Summary

- Algorithm: Logistic Regression
- Dataset: Telco Customer Churn (Kaggle)
- Training rows: 7043
- Reported accuracy: about 79%
