"""
gold_feature_engineering.py
-----------------------------------------
Purpose     :   Build ML-Ready features from Silver layer data.
Layer       :   Gold (Feature engineered, ML-Ready dataset)       
Input       :   data/silver/aml_silver.parquet
Output      :   data/gold/aml_gold.parquet
Author      :   Dhiru Misra
Version     :   1.0.0

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
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

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

SOURCE_FILE = os.path.join(PROJECT_ROOT, "data", "silver", "aml_silver.parquet")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "aml_gold.parquet")

CHUNK_SIZE = 100_000

# ==========================
# Function 1: Build amount features
# ==========================
def build_amount_features(df):
    """
    Create new numerical features from transaction amounts.
    These features capture laundering patterns in amounts.

    Args:
        df (pd.DataFrame): Silver layer dataframe
    Returns:
        pd.DataFrame: Dataframe with new amount features
    """
    # Ratio of amount received to amount paid
    # Laundering often shows unusual ratios due to currency conversion
    df["amount_ratio"] = df["amount_received"] / (df["amount_paid"] + 1e-9)

    # Log transform of amount paid — reduces impact of extreme outliers
    df["log_amount_paid"] = np.log1p(df["amount_paid"])

    # Log transform of amount received
    df["log_amount_received"] = np.log1p(df["amount_received"])

    # Absolute difference between amounts
    df["amount_difference"] = abs(df["amount_received"] - df["amount_paid"])

    logger.info("Amount features built ✅")
    return df

# ==========================
# Function 2: Encode categorical columns
# ==========================
def encode_categoricals(df):
    """
    Convert categorical text columns to numerical labels.
    ML models require numerical input — not text.

    Args:
        df (pd.DataFrame): Dataframe with amount features
    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns
    """
    categorical_columns = [
        "payment_format",
        "payment_currency",
        "receiving_currency"
    ]

    le = LabelEncoder()

    for col in categorical_columns:
        df[col] = df[col].astype(str)
        top_categories = df[col].value_counts().index.tolist()
        df[col] = df[col].apply(
            lambda x: x if x in top_categories else "Other"
        )
        df[col] = le.fit_transform(df[col])
        logger.info(f"Encoded column: {col}")
    return df

# ==========================
# Function 3: Select final features
# ==========================

def select_features(df):

    """
    Select only ML-relevant columns for gold layer.
    Drop raw columns not needed for model training.

    Args:
        df (pd.DataFrame): Fully engineered dataframe.
    Returns:
        pd.DataFrame: Dataframe with selected features only
    """
    feature_columns = [
        
        # Numerical transaction features
        "amount_received",
        "amount_paid",
        "amount_ratio",
        "log_amount_paid",
        "log_amount_received",
        "amount_difference",

        # Time features
        "hour_of_day",
        "day_of_week",

        # Bank and encoded categorical features
        "from_bank",
        "to_bank",
        "payment_format",
        "payment_currency",
        "receiving_currency",

        # Risk flags
        "is_high_risk_format",
        "is_cross_currency",
        "is_bitcoin",

        # Target variable
        "is_laundering"
    ]

    df = df[feature_columns]
    logger.info(f"Features selected: {len(feature_columns)} columns")
    logger.info(f"Feature columns: {feature_columns}")
    return df

# ==========================
# Function 4: Save Gold layer
# ==========================
def save_gold(df, output_file):
    """
    Save final ML-ready dataframe to Gold layer as Parquet.

    Args:
        df          (pd.DataFrame): Final feature engineered dataframe
        output_file (str)         : Path to save Gold parquet file
    """
    df.to_parquet(output_file, index=False)
    logger.info(f"Gold layer saved: {output_file}")
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ==========================
# Function 5: Main entry point
# ==========================

def main():
    
    """
    Main function to orchestrate Gold layer feature engineering.
    Read Silver, engineers features, saves Gold.

    """
    logger.info("=" * 60)
    logger.info("AML Gold Layer Feature Engineering - START")
    logger.info("=" * 60)

    start_time = time.time()

    # Load silver data
    logger.info("Loading Silver layer data.........")
    parquet_file = pq.ParquetFile(SOURCE_FILE)
    first_batch = next(parquet_file.iter_batches(batch_size=500_000))
    df = first_batch.to_pandas()
    logger.info(f"Loaded {len(df):,} rows from Silver layer")

    # Apply feature engineering
    df=build_amount_features(df)
    df=encode_categoricals(df)
    df=select_features(df)

    # Save Gold layer
    save_gold(df, OUTPUT_FILE)

    elapsed=time.time() - start_time
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    logger.info("=" * 60)
    logger.info("AML Gold Layer Feature Engineering — COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":
    main()