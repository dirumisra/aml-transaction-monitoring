# 🏦 AML Transaction Monitoring System

> Enterprise-grade Anti-Money Laundering detection system built with Python, Machine Learning, and Generative AI.

![CI/CD](https://github.com/dirumisra/aml-transaction-monitoring/actions/workflows/ci_dev.yml/badge.svg)

---

## 🎯 Project Overview

A production-ready AML transaction monitoring system that processes 179 million banking transactions using a full Medallion architecture pipeline, trains and compares multiple ML models, explains predictions using SHAP, and automatically generates Suspicious Activity Reports (SARs) using Generative AI.

---

## 🏗️ Architecture

Raw CSV (179M rows)
↓
Bronze Layer → chunked PyArrow ingestion
↓
Silver Layer → cleaning, feature engineering
↓
Gold Layer   → ML-ready features
↓
ML Models    → LR, Random Forest, XGBoost
↓
SHAP         → explainability
↓
GenAI        → SAR report generation
↓
Streamlit    → live dashboard

---

## 📊 Model Results

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.0008 | 0.5714 | 0.0016 | 0.9322 |
| Random Forest | 0.1154 | 0.4286 | 0.1818 | 0.8544 |
| **XGBoost** | **0.0256** | **0.4286** | **0.0484** | **0.9877** |

**Winner: XGBoost — AUC-ROC 0.9877**

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Data Processing | PyArrow, Pandas |
| ML Models | Scikit-learn, XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| GenAI | Groq (Llama 3.1) |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Architecture | Medallion (Bronze/Silver/Gold) |

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/dirumisra/aml-transaction-monitoring.git

# Create virtual environment
python -m venv venv
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/ingestion/bronze_aml_ingest.py
python src/features/silver_aml_transform.py
python src/features/gold_feature_engineering.py
python src/models/model_xgboost.py

# Launch dashboard
streamlit run streamlit_app/app.py
```

---

## 📁 Project Structure
aml-transaction-monitoring/
├── src/
│   ├── ingestion/        # Bronze layer
│   ├── features/         # Silver + Gold layers
│   ├── models/           # ML models
│   ├── evaluation/       # SHAP explainability
│   ├── dq/               # Data quality checks
│   └── genai/            # SAR report generation
├── streamlit_app/        # Dashboard
├── notebooks/            # EDA
├── reports/              # SHAP charts
├── artifacts/            # Trained models
├── docs/                 # Documentation
└── .github/workflows/    # CI/CD pipelines

---

## 👨‍💻 Author

**Dhiru Misra** — Data & AI Engineer

---

*Built as an enterprise-grade portfolio project demonstrating end-to-end ML engineering capabilities.*
```