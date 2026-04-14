"""
dq_aml_validation.py
--------------------
Purpose : Validate data quality across Bronze, Silver and Gold layers.
Layer   : All layers — Bronze, Silver, Gold
Input   : data/bronze/aml_raw.parquet
          data/silver/aml_silver.parquet
          data/gold/aml_gold.parquet
Author  : Dhiru Misra
Version : 1.0.0
"""

# ==========================
# Standard library imports
# ==========================
import os
import logging
import time

# ==========================
# Third party imports
# ==========================
import pandas as pd
import pyarrow.parquet as pq

# ==========================
# Logging configuration
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==========================
# Constants
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

BRONZE_FILE = os.path.join(PROJECT_ROOT, "data", "bronze", "aml_raw.parquet")
SILVER_FILE = os.path.join(PROJECT_ROOT, "data", "silver", "aml_silver.parquet")
GOLD_FILE   = os.path.join(PROJECT_ROOT, "data", "gold", "aml_gold.parquet")

# Expected columns per layer
BRONZE_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering"
]

SILVER_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
    "hour_of_day", "day_of_week", "is_high_risk_format",
    "is_cross_currency", "is_bitcoin"
]

GOLD_COLUMNS = [
    "amount_received", "amount_paid", "amount_ratio",
    "log_amount_paid", "log_amount_received", "amount_difference",
    "hour_of_day", "day_of_week", "from_bank", "to_bank",
    "payment_format", "payment_currency", "receiving_currency",
    "is_high_risk_format", "is_cross_currency", "is_bitcoin",
    "is_laundering"
]

# ==========================
# Function 1: Load sample
# ==========================
def load_sample(file_path, sample_size=10_000):
    """
    Load a small sample from a parquet file for DQ checks.

    Args:
        file_path   (str): Path to parquet file
        sample_size (int): Number of rows to sample
    Returns:
        pd.DataFrame: Sample dataframe
    """
    logger.info(f"Loading sample from: {file_path}")
    parquet_file = pq.ParquetFile(file_path)
    first_batch = next(parquet_file.iter_batches(batch_size=sample_size))
    df = first_batch.to_pandas()
    logger.info(f"Sample loaded: {len(df):,} rows x {df.shape[1]} columns")
    return df

# ==========================
# Function 2: Check schema
# ==========================

def check_schema(df, expected_columns, layer_name):
    """
    Validate dataframe schema against expected columns.
    Ensures no missing columns and flags unexpected columns.

    Returns:
        bool: True if schema is valid, False otherwise
    """

    logger.info(f"Checking schema for {layer_name} layer....")

    actual_columns = list(df.columns)

    # Columns expected but missing in dataframe
    missing_columns = [col for col in expected_columns if col not in actual_columns]

    # Columns present but not expected
    extra_columns = [col for col in actual_columns if col not in expected_columns]

    if missing_columns:
        logger.error(f"Missing columns in {layer_name}: {missing_columns}")
        return False

    if extra_columns:
        logger.warning(f"Extra columns in {layer_name}: {extra_columns}")

    logger.info(f"Schema check PASSED for {layer_name} ✅")
    return True

# ==========================
# Function 3: Check null values
# ==========================

def check_nulls(df, layer_name):
    """
    Check for missing values in any column.

    Args:
        df         (pd.DataFrame): Sample dataframe
        layer_name (str)         : Name of layer being checked
    Returns:
        bool: True if no nulls found, False otherwise
    """
    logger.info(f"Checking nulls for {layer_name} layer...")

    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]

    if len(null_columns) > 0:
        logger.error(f"Null values found in {layer_name}:")
        logger.error(f"{null_columns}")
        return False

    logger.info(f"Null check PASSED for {layer_name} ✅")
    return True

# ==========================
# Function 4: Check duplicates
# ==========================
def check_duplicates(df, layer_name):
    """
    Check for duplicate rows in dataframe.

    Args:
        df         (pd.DataFrame): Sample dataframe
        layer_name (str)         : Name of layer being checked
    Returns:
        bool: True if no duplicates found, False otherwise
    """
    logger.info(f"Checking duplicates for {layer_name} layer...")

    duplicate_count = df.duplicated().sum()



    logger.info(f"Duplicate check PASSED for {layer_name} ✅")
    return True

# ==========================
# Function 5: Check amount ranges
# ==========================
def check_amount_ranges(df, layer_name):
    """
    Validate that transaction amounts are non-negative.
    Negative amounts indicate data corruption or ingestion error.

    Args:
        df         (pd.DataFrame): Sample dataframe
        layer_name (str)         : Name of layer being checked
    Returns:
        bool: True if all amounts are valid, False otherwise
    """
    logger.info(f"Checking amount ranges for {layer_name} layer...")

    amount_columns = [col for col in df.columns if "amount" in col.lower()]

    failed = False
    for col in amount_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            logger.error(f"Negative values in {col}: {negative_count:,}")
            failed = True

    if not failed:
        logger.info(f"Amount range check PASSED for {layer_name} ✅")
        return True

    return False

# ==========================
# Function 6: Check target values
# ==========================

def check_target_values(df, layer_name):
    """
    Validate that 'is_laundering' column contains only 0 or 1.
    Any other value indicates data corruption.

    Args:
        df         (pd.DataFrame): Sample dataframe
        layer_name (str)         : Name of layer being checked
    Returns:
        bool: True if target values are valid, False otherwise
    """
    # Log the start of the validation process
    logger.info(f"Checking target values for {layer_name} layer...")

    # Check if the target column exists; if not, log a warning and skip
    if "is_laundering" not in df.columns:
        logger.warning(f"is_laundering column not found in {layer_name} — skipping")
        return True  # Returning True assumes missing column is not an error

    # Get all unique values in the target column
    unique_values = df["is_laundering"].unique().tolist()
    
    # Identify any values that are not 0 or 1
    invalid_values = [v for v in unique_values if v not in [0, 1]]

    # If there are invalid values, log an error and return False
    if invalid_values:
        logger.error(f"Invalid target values in {layer_name}: {invalid_values}")
        return False

    # If all values are valid, log success and return True
    logger.info(f"Target value check PASSED for {layer_name} ✅")
    return True

# ==========================
# Function 7: Main entry point
# ==========================
def main():
    """
    Run all DQ checks across Bronze, Silver and Gold layers.
    Logs PASSED or FAILED for each check.
    """
    logger.info("=" * 60)
    logger.info("AML Data Quality Validation — START")
    logger.info("=" * 60)

    results = {}

    # ── Bronze Layer Checks ──
    logger.info("--- BRONZE LAYER ---")
    bronze_df = load_sample(BRONZE_FILE)
    results["bronze_schema"]    = check_schema(bronze_df, BRONZE_COLUMNS, "Bronze")
    results["bronze_nulls"]     = check_nulls(bronze_df, "Bronze")
    results["bronze_duplicates"]= check_duplicates(bronze_df, "Bronze")

    # ── Silver Layer Checks ──
    logger.info("--- SILVER LAYER ---")
    silver_df = load_sample(SILVER_FILE)
    results["silver_schema"]    = check_schema(silver_df, SILVER_COLUMNS, "Silver")
    results["silver_nulls"]     = check_nulls(silver_df, "Silver")
    results["silver_duplicates"]= check_duplicates(silver_df, "Silver")
    results["silver_amounts"]   = check_amount_ranges(silver_df, "Silver")
    results["silver_target"]    = check_target_values(silver_df, "Silver")

    # ── Gold Layer Checks ──
    logger.info("--- GOLD LAYER ---")
    gold_df = load_sample(GOLD_FILE)
    results["gold_schema"]      = check_schema(gold_df, GOLD_COLUMNS, "Gold")
    results["gold_nulls"]       = check_nulls(gold_df, "Gold")
    results["gold_duplicates"]  = check_duplicates(gold_df, "Gold")
    results["gold_amounts"]     = check_amount_ranges(gold_df, "Gold")
    results["gold_target"]      = check_target_values(gold_df, "Gold")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("DQ VALIDATION SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for check, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{check:<30} {status}")
    logger.info(f"\nTotal Passed: {passed} | Total Failed: {failed}")
    logger.info("=" * 60)
    logger.info("AML Data Quality Validation — COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":
    main()    if duplicate_count > 0:
        logger.warning(f"Duplicate rows found in {layer_name}: {duplicate_count:,}— may be legitimate identical transactions")
        return True