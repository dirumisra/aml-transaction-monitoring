"""
silver_aml_transform.py
-----------------------
Purpose : Clean and standardize Bronze layer data into Silver layer.
Layer   : Silver (Cleaned, renamed, type-corrected data)
Input   : data/bronze/aml_raw.parquet
Output  : data/silver/aml_silver.parquet
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
import pyarrow as pa
import pyarrow.parquet as pq

# ==========================
# Logging configuration
# ==========================

import logging

# ==========================
# Logging Configuration
# ==========================
logging.basicConfig(
    level=logging.INFO,  # Set minimum logging level to INFO
    format="%(asctime)s | %(levelname)s | %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
)

# Get a logger instance
logger = logging.getLogger(__name__)  # Correct method is getLogger, not getlogger

# ==========================
# Constants
# ==========================

# Base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (2 levels up from BASE_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # Fixed syntax: "..", ".."

# Source data file (raw/bronze layer)
SOURCE_FILE = os.path.join(PROJECT_ROOT, "data", "bronze", "aml_raw.parquet")

# Output data file (processed/silver layer)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "silver", "aml_silver.parquet")

# Chunk size for processing large files
CHUNK_SIZE = 100_000

# Notes for future reference:
# - Use logger.info(), logger.warning(), etc., instead of print() for consistent logging.
# - Constants like file paths and chunk sizes are defined in uppercase by convention.
# - Adjust CHUNK_SIZE depending on memory and file size.
# - Using os.path.join ensures OS-independent file paths.

# ==========================
# Function 1: Rename columns
# ==========================

def rename_columns(df):
    
    """
    Rename all columns to snake_case:
    Fixes spaces, dots and inconsistent naming found in EDA.

    Args:
        df(pd.DataFrame): Raw Bronze dataframe
    Returns:
        pd.DataFrame: Dataframe with cleaned column names
    """
    column_mapping = {

        "Timestamp"             :   "timestamp",
        "From Bank"             :   "from_bank",
        "Account"               :   "from_account",
        "To Bank"               :   "to_bank",
        "Account.1"             :   "to_account",
        "Amount Received"       :   "amount_received",
        "Receiving Currency"    :   "receiving_currency",
        "Amount Paid"           :   "amount_paid",
        "Payment Currency"      :   "payment_currency",
        "Payment Format"        :   "payment_format",
        "Is Laundering"         :   "is_laundering"
    }

    df = df.rename(columns=column_mapping)
    logger.info(f"Columns renamed: {list(df.columns)}")
    
    return df

# ==========================
# Function 2: Fix data types
# ==========================

def fix_data_types(df):

    """
    Fix incorrect data types identified during EDA.
    Converts timestamp from string to datetime.

    Args:
        df (pd.DataFrame): Dataframe with renamed columns
    Returns:
        pd.DataFrame: DataFrame with corrected data types
    """
    # Convert timestamp from string to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")

    logger.info("Timestamp converted to datetime ✅")

    # Extract hour of day from timestamp
    df["hour_of_day"] = df["timestamp"].dt.hour
    logger.info("Hour of day extracted ✅")

    # Extract day of week from timestamp (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    logger.info("Day of week extracted ✅")

    return df

# ==========================
# Function 3: Add risk flags
# ==========================

def add_risk_flags(df):

    """
    Add binary risk flag columns based on AML domain knowledge.
    These flags will be used as features in Gold layer ML Models.

    Args:
        df (pd.DataFrame): DataFrame with fixed data types
    Returns:
        pd.DataFrame: Dataframe with added risk flag columns
    
    """
    # Flag high risk payment formats
    high_risk_formats = ["Bitcoin", "Cash", "Wire"]
    df["is_high_risk_format"] = df["payment_format"].isin(high_risk_formats).astype(int)
    logger.info("High risk payment format flag added ✅")

    # Flag cross currency transactions
    df["is_cross_currency"] = (
        df["receiving_currency"] != df["payment_currency"]
    ).astype(int)
    logger.info("Cross currency flag added ✅")

    # Flag Bitcoin transactions specifically
    df["is_bitcoin"] = (df["payment_currency"] == "Bitcoin").astype(int)
    logger.info("Bitcoin flag added ✅")

    return df

# ==========================
# Function 4: Transform Silver
# ==========================
def transform_silver(source_file, output_file, chunk_size):
    """
    Read Bronze parquet in chunks, apply all transformations,
    and save cleaned data to Silver layer.

    Args:
        source_file (str): Path to Bronze parquet file
        output_file (str): Path to save Silver parquet file
        chunk_size  (int): Number of rows per chunk
    """
    logger.info("Starting Silver layer transformation...")

    start_time = time.time()
    total_rows = 0
    writer = None

    parquet_file = pq.ParquetFile(source_file)

    for chunk_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=chunk_size), start=1
    ):
        # Convert batch to pandas dataframe
        df = batch.to_pandas()

        # Apply all transformations
        df = rename_columns(df)
        df = fix_data_types(df)
        df = add_risk_flags(df)

        # Write to parquet
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)

        total_rows += len(df)
        logger.info(f"Chunk {chunk_number} processed | Rows so far: {total_rows:,}")

    if writer:
        writer.close()

    elapsed = time.time() - start_time
    logger.info("Silver transformation complete.")
    logger.info(f"Total rows : {total_rows:,}")
    logger.info(f"Time taken : {elapsed:.2f} seconds")

# ==========================
# Function 5: Main entry point
# ==========================

def main():

    """
    Main function to orchestrate Silver layer transformation
    Calls all function in correct sequence.

    """

    logger.info("=" * 60)
    logger.info("AML Silver Layer Transformation - Start")
    logger.info("=" * 60)

    transform_silver(SOURCE_FILE, OUTPUT_FILE, CHUNK_SIZE)

    logger.info("=" * 60)
    logger.info("AML Silver Layer Transformation - COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================

if __name__ == "__main__":
    main()