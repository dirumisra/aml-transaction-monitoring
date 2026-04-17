"""
02_model_performance.py
------------------------
Purpose : Model performance page — metrics and SHAP explainability.
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image  # Used for image handling (optional if further processing needed)

# ----------------------------
# Page configuration
# ----------------------------

# Set Streamlit page settings
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

# Page title and separator
st.title("📈 Model Performance")
st.markdown("---")


# ==========================
# Model comparison table
# ==========================

# Section header
st.subheader("🏆 Model Comparison")

# Static model evaluation metrics
# NOTE: These values are precomputed and hardcoded
metrics_data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Precision": [0.0008, 0.1154, 0.0256],
    "Recall": [0.5714, 0.4286, 0.4286],
    "F1 Score": [0.0016, 0.1818, 0.0484],
    "AUC-ROC": [0.9322, 0.8544, 0.9877],
    "Winner": ["", "", "✅"]  # Indicates best performing model
}

# Convert dictionary to DataFrame
df = pd.DataFrame(metrics_data)

# Display model comparison table
st.dataframe(df, use_container_width=True)

st.markdown("---")


# ==========================
# SHAP charts
# ==========================

# Section header for explainability visuals
st.subheader("🔍 SHAP Explainability")

# ----------------------------
# File path setup
# ----------------------------

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate to project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Create two columns for side-by-side visualizations
col1, col2 = st.columns(2)

with col1:
    # SHAP summary plot (global feature importance)
    st.markdown("**Feature Importance — Summary Plot**")
    
    # Path to summary plot image
    summary_path = os.path.join(PROJECT_ROOT, "reports", "shap_summary.png")
    
    # Display image
    st.image(summary_path, use_column_width=True)

with col2:
    # SHAP waterfall plot (individual prediction explanation)
    st.markdown("**Single Transaction — Waterfall Plot**")
    
    # Path to waterfall plot image
    waterfall_path = os.path.join(PROJECT_ROOT, "reports", "shap_waterfall.png")
    
    # Display image
    st.image(waterfall_path, use_column_width=True)