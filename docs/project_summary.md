# Project Summary — AML Transaction Monitoring System

**Author:** Dhiru Misra
**Date:** April 2026
**Status:** Complete

---

## Problem Statement

Money laundering costs the global economy $2 trillion annually. Traditional rule-based AML systems generate excessive false positives and require manual SAR writing — taking compliance officers 30-60 minutes per report.

---

## Solution Built

An end-to-end enterprise AML system that:
- Processes 179 million transactions automatically
- Detects suspicious transactions with AUC-ROC of 0.9877
- Explains every flag using SHAP values
- Generates professional SAR reports in under 2 seconds using AI

---

## Key Achievements

| Achievement | Detail |
|---|---|
| Data processed | 179,702,225 transactions |
| Best model AUC | 0.9877 (XGBoost) |
| SAR generation time | Under 2 seconds |
| CI/CD pipelines | 3 automated pipelines |
| Dashboard pages | 3 live Streamlit pages |
| Git branches | DEV → UAT → MAIN |

---

## Technical Highlights

- **Medallion architecture** — Bronze, Silver, Gold data layers
- **Memory-safe ingestion** — 179M rows without overflow using PyArrow chunking
- **3 ML models compared** — LR, Random Forest, XGBoost via MLflow
- **SHAP explainability** — regulatory compliance ready
- **GenAI SAR generation** — Groq Llama 3.1 integration
- **Enterprise CI/CD** — GitHub Actions on 3 branches

---

## Business Impact