# Technical Design Document — AML Transaction Monitoring System

**Version:** 1.0.0
**Author:** Dhiru Misra
**Date:** April 2026

---

## 1. System Overview

End-to-end AML transaction monitoring system processing 179 million banking transactions through a Medallion architecture pipeline, ML model ensemble, SHAP explainability, and GenAI-powered SAR generation.

---

## 2. Architecture — Medallion Pipeline

### Bronze Layer
- **Script:** `src/ingestion/bronze_aml_ingest.py`
- **Input:** HI-Large_Trans.csv (179M rows)
- **Output:** `data/bronze/aml_raw.parquet`
- **Key decision:** Chunked PyArrow ingestion (100K rows/chunk) — prevents memory overflow on 179M rows

### Silver Layer
- **Script:** `src/features/silver_aml_transform.py`
- **Input:** Bronze parquet
- **Output:** `data/silver/aml_silver.parquet`
- **Transformations:** Column renaming, timestamp parsing, risk flags (is_bitcoin, is_cross_currency, is_high_risk_format)

### Gold Layer
- **Script:** `src/features/gold_feature_engineering.py`
- **Input:** Silver parquet
- **Output:** `data/gold/aml_gold.parquet`
- **Features:** amount_ratio, log_amount_paid, log_amount_received, LabelEncoded categoricals

---

## 3. ML Pipeline

### Data Split
- 80% training / 20% test
- Stratified split — maintains fraud ratio in both sets

### Class Imbalance Strategy
- SMOTE applied on training data only
- scale_pos_weight=14285 in XGBoost
- class_weight=balanced in LR and RF

### Model Selection
Three models trained and compared via MLflow:
- Logistic Regression → baseline
- Random Forest → bagging ensemble
- XGBoost → boosting ensemble → selected

### Experiment Tracking
- MLflow experiment: `aml-fraud-detection`
- All runs tracked with params, metrics, artifacts

---

## 4. Explainability — SHAP

- TreeExplainer used for XGBoost compatibility
- Summary plot → global feature importance
- Waterfall plot → per-transaction explanation
- Required for AML regulatory compliance

---

## 5. GenAI — SAR Generation

- **Provider:** Groq (free tier)
- **Model:** llama-3.1-8b-instant
- **Input:** Transaction details + ML confidence score
- **Output:** Formal SAR report (BSA/PATRIOT Act compliant)
- **Temperature:** 0.3 — consistent formal output

---

## 6. Dashboard — Streamlit

Three pages:
1. Transaction Monitoring — filter and view transacti