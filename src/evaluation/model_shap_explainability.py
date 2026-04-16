"""
model_shap_explainability.py
-----------------------------
Purpose : Explain XGBoost model predictions using SHAP values.
Layer   : Model Evaluation — Phase 4
Input   : artifacts/xgboost.pkl
          data/gold/aml_gold.parquet
Output  : reports/shap_summary.png
          reports/shap_waterfall.png
Author  : Dhiru Misra
Version : 1.0.0
"""

# ==========================
# Standard library imports
# ==========================
import os
import logging

# ==========================
# Third party imports

# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

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

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the root of the project by going up two levels from the script directory
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Path to the serialized machine learning model (XGBoost)
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "xgboost.pkl")

# Path to the "gold" dataset (preprocessed or ground truth data)
GOLD_FILE   = os.path.join(PROJECT_ROOT, "data", "gold", "aml_gold.parquet")

# Directory where generated reports will be saved
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

# Target column for model prediction
TARGET_COLUMN = "is_laundering"

# Number of sample rows to work with (for testing or quick runs)
SAMPLE_SIZE    = 1000

# ==========================
# Function 1: Load model and data
# ==========================
def load_model_and_data():
    """
    Load trained XGBoost model and Gold layer data.
    Sample 1000 rows for SHAP computation.

    Returns:
        model    : Trained XGBoost model
        X_sample : Sample feature dataframe
    """
    logger.info("Loading XGBoost model...")  # Log the start of model loading
    model = joblib.load(MODEL_PATH)  # Load the pre-trained XGBoost model from the specified path
    logger.info("Model loaded ✅")  # Log successful model loading

    logger.info("Loading Gold layer data...")  # Log the start of data loading
    df = pd.read_parquet(GOLD_FILE)  # Read the data from the Gold layer file (parquet format)

    # Separate features from target — drop the target column to get only features
    X = df.drop(columns=[TARGET_COLUMN])  # X will hold the feature columns, excluding the target

    # Sample for SHAP computation — using a subset of data to speed up SHAP calculations (full dataset is too slow)
    X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)  # Randomly sample n rows from the dataset
    logger.info(f"Sample loaded: {X_sample.shape[0]:,} rows x {X_sample.shape[1]} columns")  # Log the size of the sample

    return model, X_sample  # Return the loaded model and the sampled feature dataframe

# ==========================
# Function 2: Compute SHAP values
# ==========================
def compute_shap_values(model, X_sample):
    """
    Compute SHAP values for the sample dataset.
    SHAP values explain how each feature contributes to predictions.

    Args:
        model    : Trained XGBoost model
        X_sample : Sample feature dataframe
    Returns:
        explainer   : SHAP explainer object
        shap_values : SHAP values array
    """
    logger.info("Computing SHAP values...")

    # Create TreeExplainer — optimised for tree based models like XGBoost
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for all sample rows
    shap_values = explainer.shap_values(X_sample)

    logger.info(f"SHAP values computed ✅")
    logger.info(f"SHAP values shape: {shap_values.shape}")

    return explainer, shap_values

# ==========================
# Function 3: Plot SHAP summary
# ==========================
def plot_summary(shap_values, X_sample):
    """
    Generate SHAP summary bar plot.
    Shows which features matter most across all predictions.

    Args:
        shap_values : SHAP values array
        X_sample    : Sample feature dataframe
    """
    # Log the start of the plot generation process
    logger.info("Generating SHAP summary plot...")

    # Create a new figure for the plot with specified size (10 inches by 8 inches)
    plt.figure(figsize=(10, 8))

    # Generate the SHAP summary plot using a bar chart
    # `shap.summary_plot()` will use the SHAP values and feature dataset
    # The `plot_type="bar"` argument specifies that we want a bar chart
    # `show=False` means the plot won't be shown interactively, just saved
    shap.summary_plot(
        shap_values,   # The SHAP values that explain the model's predictions
        X_sample,      # The sample dataset for which SHAP values are computed
        plot_type="bar",  # Create a bar chart to summarize feature importance
        show=False         # Do not display the plot interactively
    )

    # Define the output path where the plot will be saved (in the reports folder)
    output_path = os.path.join(REPORTS_DIR, "shap_summary.png")

    # Save the plot as a PNG file at the specified output path
    # `bbox_inches="tight"` ensures the plot is tightly cropped with minimal padding
    # `dpi=150` specifies the resolution of the saved plot (150 dots per inch)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)

    # Close the plot to release resources (important if generating multiple plots)
    plt.close()

    # Log the successful saving of the SHAP summary plot, including the output path
    logger.info(f"SHAP summary plot saved: {output_path} ✅")

# ==========================
# Function 4: Plot SHAP waterfall
# ==========================

def plot_waterfall(explainer, shap_values, X_sample):
    """
    Generate SHAP waterfall plot for a single transaction.
    Shows exactly why one specific transaction was flagged.

    Args:
        explainer   : SHAP explainer object
        shap_values : SHAP values array
        X_sample    : Sample feature data frame
    """
    # Log the start of the waterfall plot generation
    logger.info("Generating SHAP waterfall plot...")

    # Create a new figure with a specified size (10 inches by 8 inches)
    plt.figure(figsize=(10, 8))

    # Take the first row of shap_values (explanation for the first transaction)
    # `shap.waterfall_plot()` creates a waterfall plot for the specific transaction
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],              # SHAP values for the first row
            base_values=explainer.expected_value,  # Base value (average model output)
            data=X_sample.iloc[0],               # Data (features) for the first row
            feature_names=X_sample.columns.tolist()  # Feature names for labeling
        ),
        show=False  # Do not show the plot interactively
    )

    # Define the output path where the waterfall plot will be saved
    output_path = os.path.join(REPORTS_DIR, "shap_waterfall.png")

    # Save the waterfall plot as a PNG file to the specified output path
    # `bbox_inches="tight"` ensures the plot is tightly cropped
    # `dpi=150` sets the resolution of the saved plot
    plt.savefig(output_path, bbox_inches="tight", dpi=150)

    # Close the plot to free up memory (important if generating multiple plots)
    plt.close()

    # Log the successful saving of the SHAP waterfall plot
    logger.info(f"SHAP waterfall plot saved: {output_path} ✅")

# ==========================
# Function 5: Main entry point
# ==========================

def main():
    """
    Orchestrate full SHAP explainability pipeline.
    Load → Compute → Plot Summary → Plot Waterfall
    """

    logger.info("=" * 60)
    logger.info("SHAP Explainability Pipeline — START")
    logger.info("=" * 60)

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Step 1: Load model and data
    model, X_sample = load_model_and_data()

    # Step 2: Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X_sample)

    # Step 3: Generate summary plot
    plot_summary(shap_values, X_sample)

    # Step 4: Generate waterfall plot
    plot_waterfall(explainer, shap_values, X_sample)

    logger.info("=" * 60)
    logger.info("SHAP Explainability Pipeline — COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Reports saved to: {REPORTS_DIR}")

# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":
    main()