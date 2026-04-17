"""
app.py
------
Purpose : Main entry point for AML Transaction Monitoring Dashboard.
Layer   : Streamlit Dashboard — Phase 6
Author  : Dhiru Misra
Version : 1.0.0
"""

import streamlit as st

# ==========================
# Page configuration
# ==========================
st.set_page_config(
    page_title="AML Transaction Monitoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================
# Main page content
# ==========================
st.title("🏦 AML Transaction Monitoring System")
st.markdown("---")

st.markdown("""
### Enterprise-Grade Anti-Money Laundering Detection
This Dashboard provides real-time monitoring and analysis of banking transactions
using Machine Learning and Generative AI.

---

### 📊 Navigation Guide

| Page | Description |
|---|---|
| 🔍 Transaction Monitoring | View and filter suspicious transactions |
| 📈 Model Performance | ML model metrics and SHAP explainability |
| 📋 SAR Reports | Generate AI-powered Suspicious Activity Reports |

---
### 🎯 Key Statistics
""")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "500,000")

with col2:
    st.metric("Suspicious Flagged", "35")

with col3:
    st.metric("Best Model AUC", "0.9877")

with col4:
    st.metric("SAR Reports Generated", "1")

st.markdown("---")
st.markdown("**Built by Dhiru Misra** | AML Transaction Monitoring System v1.0.0")
