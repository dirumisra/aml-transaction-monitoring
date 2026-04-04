"""
bronze_aml_ingest.py
--------------------
Purpose : Ingest raw IBM AML transaction data from CSV into Bronze layer.
Layer   : Bronze (Raw ingestion — no transformations applied)
Input   : data/bronze/HI-Large_Trans.csv
Output  : data/bronze/aml_raw.parquet
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
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level (INFO and above will be shown)
    format="%(asctime)s | %(levelname)s | %(message)s",  # Log message format
    datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
)

# Create a logger for this module/file
# __name__ helps identify which file the log is coming from
logger = logging.getLogger(__name__)

# ==========================
# Constants
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

SOURCE_FILE = os.path.join(PROJECT_ROOT, "data", "bronze", "HI-Large_Trans.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"data", "bronze", "aml_raw.parquet")

CHUNK_SIZE = 100_000

# ==========================
# Function 1: Validate source file
# ==========================

def validate_source_file(file_path):
    """    
    check if the source CSV file exists.
    Raise FileNotFoundError if file is missing.

    Args:
        file_path (str): Full path to the source CSV file
    """
    logger.info(f"Validating source file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Source file not found: {file_path}")
        raise FileNotFoundError(f"Source file not found: {file_path}")

    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    logger.info(f"Source file found. Size: {file_size_gb:.2f} GB")

# ==========================
# Function 2: Peek at raw data
# ==========================

def peek_at_data(file_path):
    """
    Read first 5 rows of CSV to preview structure.
    Logs column names, shape, and data types.

    Args:
        file_path (str): Full path to the source CSV file
    """
    logger.info("Peeking at raw data structure...")

    # Read only the first 5 rows of the CSV
    sample = pd.read_csv(file_path, nrows=5)

    # Log information about the data
    logger.info(f"Columns    : {list(sample.columns)}")
    logger.info(f"Shape      : {sample.shape}")
    logger.info(f"Data types :\n{sample.dtypes}")

    # Return column names as a list
    return list(sample.columns)

# ==========================
# Function 3: Ingest raw data into Bronze layer
# ==========================
def ingest_bronze(source_file, output_file, chunk_size):
    """
    Read CSV in chunks and save directly to Parquet.
    Each chunk is written immediately - No memory overflow.

    Args:
        source_file (str): Path to source CSV file
        output_file (str): Path to save output Parquet file
        chunk_size (int) : Number of rows to read per chunk
    """
    start_time = time.time()
    total_rows = 0
    writer = None

    # Read and write chunk by chunk
    for chunk_number, chunk in enumerate(pd.read_csv(source_file, chunksize=chunk_size), start=1):
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)
        total_rows += len(chunk)
        logger.info(f"Chunk {chunk_number} written | Rows so far: {total_rows:,}")

    if writer:
        writer.close()

    elapsed = time.time() - start_time
    logger.info("Bronze ingestion complete.")
    logger.info(f"Total rows : {total_rows:,}")
    logger.info(f"Time taken : {elapsed:.2f} seconds")

# ==========================
# Function 4: Main entry point
# ==========================

def main():

    """
    Main function to orchestrate Bronze layer ingestion.
    Call all function in correct sequence.
    
    """
    logger.info("=" * 60)
    logger.info("AML Bronze Layer Ingestion - START")
    logger.info("=" * 60)

    # Step 1: Validate source file exists
    validate_source_file(SOURCE_FILE)

    # Step 2: Peek at data structure
    peek_at_data(SOURCE_FILE)

    # Step 3: Ingest full dataset into Bronze layer
    ingest_bronze(SOURCE_FILE, OUTPUT_FILE, CHUNK_SIZE)

    logger.info("=" * 60)
    logger.info("AML Bronze Layer Ingestion - COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================

if __name__ == "__main__":
    main()