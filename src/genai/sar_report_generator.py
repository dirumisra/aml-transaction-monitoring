"""
sar_report_generator.py
-----------------------
Purpose : Generate Suspicious Activity Reports (SAR) using Groq LLM.
Layer   : GenAI — Phase 5
Input   : Suspicious transaction details + model confidence score
Output  : reports/sar_report.txt
Author  : Dhiru Misra
Version : 1.0.0
"""

# ==========================
# Standard library imports
# ==========================
import os
import logging
from datetime import datetime

# ==========================
# Third party imports
# ==========================
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================
# Logging configuration
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger  = logging.getLogger(__name__)

# ==========================
# Constants
# ==========================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(BASE_DIR,"..",".."))
REPORTS_DIR     = os.path.join(PROJECT_ROOT,"reports")

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
MODEL_NAME      = "llama-3.1-8b-instant"

# ==========================
# Function 1: Build Suspicious Activity (SAR) Report  Prompt
# ==========================
def build_prompt(transaction: dict, confidence_score: float) -> str:
    """
    Build a structured prompt for SAR report generation.

    Args:
        transaction      : Dictionary of transaction details
        confidence_score : ML model fraud probability score
    Returns:
        str: Formatted prompt for LLM
    """
    prompt = f"""
You are a senior AML compliance officer at a global bank.
Generate a professional Suspicious Activity Report (SAR) based on the following flagged transaction.

TRANSACTION DETAILS:
- Timestamp          : {transaction.get('timestamp', 'N/A')}
- From Bank          : {transaction.get('from_bank', 'N/A')}
- From Account       : {transaction.get('from_account', 'N/A')}
- To Bank            : {transaction.get('to_bank', 'N/A')}
- To Account         : {transaction.get('to_account', 'N/A')}
- Amount Paid        : {transaction.get('amount_paid', 'N/A')}
- Payment Currency   : {transaction.get('payment_currency', 'N/A')}
- Amount Received    : {transaction.get('amount_received', 'N/A')}
- Receiving Currency : {transaction.get('receiving_currency', 'N/A')}
- Payment Format     : {transaction.get('payment_format', 'N/A')}
- ML Confidence Score: {confidence_score:.2%}

Generate a SAR report with these sections:
1. Executive Summary
2. Transaction Description
3. Risk Indicators Identified
4. Recommended Action

Keep it professional, concise and regulatory compliant.
"""
    return prompt

# ==========================
# Function 2: Generate Suspicious Activity (SAR) Report using Groq
# ==========================
def generate_sar_report(transaction: dict, confidence_score: float) -> str:
    """
    Generates a Suspicious Activity Report (SAR) using Groq LLM.

    Args:
        transaction (dict): Dictionary containing transaction details such as amount, sender, receiver, etc.
        confidence_score (float): Fraud probability score predicted by the ML model.

    Returns:
        str: Generated SAR report in formal compliance language.
    """

    # Log the start of Groq client initialization
    logger.info("Initialising Groq client...")

    # Create Groq client instance using API key for authentication
    client = Groq(api_key=GROQ_API_KEY)

    # Build the LLM prompt using transaction data and model score
    prompt = build_prompt(transaction, confidence_score)
    logger.info("Prompt built successfully")

    # Log API call details for traceability/debugging
    logger.info(f"Calling Groq API — model: {MODEL_NAME}")

    # Send request to Groq Chat Completion API
    response = client.chat.completions.create(
        model=MODEL_NAME,

        # Define conversation messages for the LLM
        messages=[
            {
                # System message sets behavior/persona of the model
                "role": "system",
                "content": "You are a professional AML compliance officer. Generate formal SAR reports."
            },
            {
                # User message contains the actual prompt with transaction details
                "role": "user",
                "content": prompt
            }
        ],

        # Lower temperature = more deterministic, formal output (good for compliance reports)
        temperature=0.3,

        # Limit response length to avoid overly long outputs
        max_tokens=1000
    )

    # Extract generated SAR report text from API response
    report = response.choices[0].message.content

    # Log successful report generation
    logger.info("SAR report generated successfully ✅")

    # Return final SAR report
    return report

# ==========================
# Function 3: Save Suspicious Activity (SAR) Report report
# ==========================

def save_report(report: str, transaction_id: str) -> str:
    """
    Save generated SAR report to reports folder as a text file.

    Args:
        report (str): Generated SAR report text
        transaction_id (str): Unique identifier for the transaction

    Returns:
        str: File path where the report was saved
    """

    # Ensure the reports directory exists (creates it if missing)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Generate a unique filename using transaction ID + current timestamp
    # Format: SAR_<transaction_id>_<YYYYMMDD_HHMMSS>.txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SAR_{transaction_id}_{timestamp}.txt"

    # Construct full file path
    output_path = os.path.join(REPORTS_DIR, filename)

    # Write the SAR report content to file
    # Use 'w' mode to create/overwrite file in text mode
    with open(output_path, "w") as f:
        f.write(report)

    # Log success message with saved file location
    logger.info(f"SAR report saved: {output_path} ✅")

    # Return the file path for downstream use (e.g., audit, download, API response)
    return output_path

# ==========================
# Function 4: Main entry point
# ==========================

def main():
    """
    End-to-end SAR generation pipeline.

    Flow:
    1. Define a suspicious transaction (input)
    2. Assign ML confidence score
    3. Generate SAR report using LLM
    4. Display report
    5. Save report to file
    """

    # ==========================
    # Pipeline Start Logging
    # ==========================
    logger.info("=" * 60)
    logger.info("SAR Report Generation Pipeline - START")
    logger.info("=" * 60)

    # ==========================
    # Step 1: Prepare Input Data
    # ==========================
    # Sample suspicious transaction (simulates flagged fraud case)
    sample_transaction = {
        "timestamp"          : "2022-09-26 03:14:00",
        "from_bank"          : "20",
        "from_account"       : "800482480",   # FIXED key (was 'from account')
        "to_bank"            : "3503",
        "to_account"         : "8016C2920",
        "amount_paid"        : 850000.00,
        "payment_currency"   : "Bitcoin",
        "amount_received"    : 850000.00,
        "receiving_currency" : "Bitcoin",
        "payment_format"     : "Bitcoin"
    }

    # ==========================
    # Step 2: Model Output Input
    # ==========================
    # ML model confidence score (fraud probability)
    confidence_score = 0.92

    # ==========================
    # Step 3: Generate SAR Report (GenAI Layer)
    # ==========================
    # Calls Groq LLM to convert structured data → compliance narrative
    report = generate_sar_report(sample_transaction, confidence_score)

    # ==========================
    # Step 4: Display Output
    # ==========================
    # Print generated SAR report for quick inspection/debugging
    print("\n" + "=" * 60)
    print("GENERATED SAR REPORT")
    print("=" * 60)
    print(report)

    # ==========================
    # Step 5: Persist Output
    # ==========================
    # Save report to local filesystem for audit/compliance records
    output_path = save_report(report, transaction_id="TXN_SAMPLE_001")

    # ==========================
    # Pipeline Completion Logging
    # ==========================
    logger.info(f"Report saved at: {output_path}")
    logger.info("=" * 60)
    logger.info("SAR Report Generation Pipeline - COMPLETE")  # FIXED typo
    logger.info("=" * 60)


# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":   # FIXED syntax
    main()    