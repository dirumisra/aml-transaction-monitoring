"""
03_sar_reports.py
-----------------
Purpose : SAR report generation page — AI powered compliance reports.
"""

import streamlit as st
import sys
import os

# ----------------------------
# Page configuration
# ----------------------------

# Configure Streamlit page settings
st.set_page_config(page_title="SAR Reports", page_icon="📋", layout="wide")

# Page title and separator
st.title("📋 SAR Report Generator")
st.markdown("---")


# ----------------------------
# Project path setup
# ----------------------------

# Add project root to Python path so custom modules can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# Import SAR generation utilities (LLM-powered)
from src.genai.sar_report_generator import generate_sar_report, save_report


# ----------------------------
# User input section
# ----------------------------

st.subheader("🔍 Enter Transaction Details")

# Create two-column layout for form inputs
col1, col2 = st.columns(2)

with col1:
    # Sender details
    from_bank    = st.text_input("From Bank", value="20")
    from_account = st.text_input("From Account", value="800482480")
    
    # Transaction amount sent
    amount_paid  = st.number_input("Amount Paid", value=850000.00)
    
    # Currency used for payment
    payment_currency = st.selectbox(
        "Payment Currency",
        ["Bitcoin", "USD", "EUR", "GBP"]
    )

with col2:
    # Receiver details
    to_bank      = st.text_input("To Bank", value="3503")
    to_account   = st.text_input("To Account", value="8016C2920")
    
    # Transaction amount received
    amount_received = st.number_input("Amount Received", value=850000.00)
    
    # Mode of transaction
    payment_format  = st.selectbox(
        "Payment Format",
        ["Bitcoin", "Wire", "Credit Card", "Cash"]
    )

# ML model confidence score for suspicious activity
confidence_score = st.slider(
    "ML Confidence Score",
    0.0, 1.0, 0.92
)

st.markdown("---")


# ----------------------------
# SAR report generation
# ----------------------------

# Trigger report generation on button click
if st.button("🚀 Generate SAR Report"):

    # Construct transaction dictionary from user inputs
    transaction = {
        "timestamp"          : "2022-09-26 03:14:00",  # Static timestamp (can be dynamic)
        "from_bank"          : from_bank,
        "from_account"       : from_account,
        "to_bank"            : to_bank,
        "to_account"         : to_account,
        "amount_paid"        : amount_paid,
        "payment_currency"   : payment_currency,
        "amount_received"    : amount_received,
        "receiving_currency" : payment_currency,  # Assuming same currency
        "payment_format"     : payment_format
    }

    # Show loading spinner while generating report using LLM
    with st.spinner("Generating SAR report using Groq LLM..."):
        
        # Generate narrative SAR report
        report = generate_sar_report(transaction, confidence_score)
        
        # Save report to file/system
        output_path = save_report(report, "TXN_DASHBOARD")

    # Success message
    st.success("SAR Report Generated ✅")

    # Display generated report
    st.markdown("### Generated Report")
    st.markdown(report)

    # ----------------------------
    # Download option
    # ----------------------------

    # Allow user to download SAR report as text file
    st.download_button(
        label="📥 Download SAR Report",
        data=report,
        file_name="SAR_Report.txt",
        mime="text/plain"
    )