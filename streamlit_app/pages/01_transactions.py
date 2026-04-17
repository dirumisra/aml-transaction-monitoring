"""
01_transactions.py
------------------
Purpose : Transaction monitoring page — view and filter suspicious transactions.
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import os

# Configure Streamlit page settings (title, icon, layout)
st.set_page_config(page_title="Transaction Monitoring", page_icon="🔍", layout="wide")

# Page title and separator
st.title("🔍 Transaction Monitoring")
st.markdown("---")

# ----------------------------
# File path setup
# ----------------------------

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate to project root (two levels up)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Define path to the processed (gold layer) dataset
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "aml_gold.parquet")


# ----------------------------
# Data loading (cached)
# ----------------------------

@st.cache_data
def load_data():
    """
    Load a subset of parquet data for performance optimization.
    Reads only the first batch (10k rows) to keep UI responsive.
    """
    pf = pq.ParquetFile(GOLD_FILE)
    batch = next(pf.iter_batches(batch_size=10000))
    return batch.to_pandas()

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
    # Total number of rows displayed
    st.metric("Total Rows", f"{len(df):,}")

with col2:
    # Total number of suspicious transactions
    st.metric("Suspicious", f"{df['is_laundering'].sum():,}")