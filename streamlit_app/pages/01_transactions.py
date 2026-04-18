"""
01_transactions.py
------------------
Purpose : Transaction monitoring page — view and filter suspicious transactions.
"""

import streamlit as st
import pandas as pd
import os

# Configure Streamlit page settings (title, icon, layout)
st.set_page_config(page_title="Transaction Monitoring", page_icon="🔍", layout="wide")

# Page title and separator
st.title("🔍 Transaction Monitoring")
st.markdown("---")

# ----------------------------
# File path setup
# ----------------------------

# Get current file directory (pages/ folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up to streamlit_app/ folder
STREAMLIT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Sample data lives in streamlit_app/ — committed to GitHub
SAMPLE_FILE = os.path.join(STREAMLIT_DIR, "sample_data.csv")

# ----------------------------
# Data loading (cached)
# ----------------------------

@st.cache_data
def load_data():
    """
    Load sample data from CSV file.
    500 rows committed to GitHub for Streamlit Cloud compatibility.
    Full 179M row dataset available locally via Gold parquet layer.
    """
    return pd.read_csv(SAMPLE_FILE)

# Load dataframe
df = load_data()

# ----------------------------
# Sidebar filters
# ----------------------------

st.sidebar.header("Filters")

# Checkbox to filter only suspicious transactions
show_suspicious = st.sidebar.checkbox("Show Suspicious Only", value=False)

# Apply filter if checkbox is selected
if show_suspicious:
    df = df[df["is_laundering"] == 1]

# ----------------------------
# Main table display
# ----------------------------

# Display number of records currently shown
st.markdown(f"### Showing {len(df):,} transactions")

# Render dataframe in full width container
st.dataframe(df, use_container_width=True)

# ----------------------------
# Summary metrics
# ----------------------------

# Create two columns for metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Rows", f"{len(df):,}")

with col2:
    st.metric("Suspicious", f"{df['is_laundering'].sum():,}")