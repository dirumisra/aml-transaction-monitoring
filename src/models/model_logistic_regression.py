"""
model_logistic_regression.py
-----------------------------
Purpose : Train and evaluate Logistic Regression model on Gold layer data.
Layer   : ML Models — Phase 3
Input   : data/gold/aml_gold.parquet
Output  : artifacts/logistic_regression.pkl
Author  : Dhiru Misra
Version : 1.0.0

"""

# ==========================
# Standard library imports
# ==========================

import os
import logging
import time
import joblib
import mlflow
import mlflow.sklearn

# ==========================
# Third party imports
# ==========================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

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
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..",".."))

GOLD_FILE     =  os.path.join(PROJECT_ROOT, "data","gold","aml_gold.parquet")
ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "logistic_regression.pkl")

TARGET_COLUMN = "is_laundering"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# ==========================
# Function 1: Load Gold data
# ==========================

def load_data(file_path):
    """
    Load Gold layer parquet file for model training.

    Args:
        file_path (str): Path to gold parquet file
    Returns:
        pd.DataFrame: Gold layer dataframe

    """

    logger.info("Loading Gold layer data....")
    df = pd.read_parquet(file_path)
    logger.info(f"Data loaded: {df.shape[0]:,} rows * {df.shape[1]} columns")
    logger.info(f"Target distribution:\n{df[TARGET_COLUMN].value_counts()}")

    return df

# ==========================
# Function 2: Prepare data
# ==========================
def prepare_data(df):
    """
    Split dataframe into features and target.
    Apply 80/20 train test split.

    Args:
        df (pd.DataFrame): Gold layer dataframe
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing data for training...")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    logger.info(f"Features shape : {X.shape}")
    logger.info(f"Target shape   : {y.shape}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Training set   : {X_train.shape[0]:,} rows")
    logger.info(f"Test set       : {X_test.shape[0]:,} rows")

    return X_train, X_test, y_train, y_test

# ==========================
# Function 3: Apply SMOTE
# ==========================
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance minority class in training data.
    SMOTE is applied ONLY on training data — never on test data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series)   : Training labels
    Returns:
        X_resampled, y_resampled: Balanced training data
    """

    # Step 1: Print message that SMOTE process is starting
    logger.info("Applying SMOTE to balance classes...")

    # Step 2: Show count of each class BEFORE SMOTE (e.g., 0=9500, 1=500)
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")

    # Step 3: Create SMOTE object (random_state ensures same result every time)
    smote = SMOTE(random_state=RANDOM_STATE)

    # Step 4: Apply SMOTE
    # It creates new synthetic rows for minority class (1)
    # Output: balanced X and y
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Step 5: Show count of each class AFTER SMOTE (now balanced)
    logger.info(f"After SMOTE : {pd.Series(y_resampled).value_counts().to_dict()}")

    # Step 6: Show new total number of rows after adding synthetic data
    logger.info(f"Resampled training set: {X_resampled.shape[0]:,} rows")

    # Step 7: Return balanced dataset
    return X_resampled, y_resampled

# ==========================
# Function 4: Train model
# ==========================
def train_model(X_train, y_train):
    """
    Train Logistic Regression model on SMOTE balanced data.

    Args:
        X_train (pd.DataFrame): Balanced training features
        y_train (pd.Series)   : Balanced training labels
    Returns:
        LogisticRegression: Trained model
    """
    # Step 1: Log that training is starting
    logger.info("Training Logistic Regression model...")

    # Step 2: Create model object with parameters
    model = LogisticRegression(
        max_iter=1000,            # Max number of iterations for model to learn (avoid convergence error)
        random_state=RANDOM_STATE, # Ensures same results every run
        class_weight="balanced"   # Automatically gives more importance to minority class
    )
 
    # Step 3: Train the model using training data
    # Model learns relationship between X (features) and y (target)
    model.fit(X_train, y_train)

    # Step 4: Log that training is completed
    logger.info("Model training complete ✅")

    # Step 5: Return trained model (ready for prediction)
    return model

# ==========================
# Function 5: Evaluate model
# ==========================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model using multiple metrics.
    Uses test data only — model has never seen this data.

    Args:
        model  : Trained model
        X_test : Test features (input)
        y_test : Test labels (actual output)
    Returns:
        dict: Dictionary of evaluation metrics
    """

    # Step 1: Print that evaluation is starting
    logger.info("Evaluating model on test data...")

    # Step 2: Get final predictions (0 or 1)
    y_pred = model.predict(X_test)

    # Step 3: Get probability of class 1 (fraud confidence)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Step 4: Calculate Precision
    # Out of predicted fraud → how many are correct
    precision = precision_score(y_test, y_pred, zero_division=0)

    # Step 5: Calculate Recall
    # Out of actual fraud → how many we caught
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Step 6: Calculate F1 Score
    # Balance between precision and recall
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Step 7: Calculate ROC-AUC
    # Checks how well model ranks fraud vs normal using probability
    auc_roc = roc_auc_score(y_test, y_prob)

    # Step 8: Create Confusion Matrix
    # Shows TP, TN, FP, FN in table form
    cm = confusion_matrix(y_test, y_pred)

    # Step 9: Print results nicely
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)

    # Step 10: Print all metrics
    logger.info(f"Precision : {precision:.4f}")  # avoid false alarms
    logger.info(f"Recall    : {recall:.4f}")    # catch fraud
    logger.info(f"F1 Score  : {f1:.4f}")        # balance both
    logger.info(f"AUC-ROC   : {auc_roc:.4f}")   # ranking quality

    # Step 11: Print confusion matrix
    logger.info(f"Confusion Matrix:\n{cm}")

    # Step 12: Detailed report (precision, recall, f1 per class)
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Step 13: Return metrics as dictionary
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc
    }

# ==========================
# Function 6: Save model
# ==========================
def save_model(model, artifact_path):
    """
    Save trained model to disk as pkl file.
    Model can be loaded later without retraining.

    Args:
        model         : Trained model object
        artifact_path : Path to save pkl file
    """

    # Step 1: Save model into a file (.pkl format)
    # joblib converts model object → file and stores on disk
    joblib.dump(model, artifact_path)

    # Step 2: Log message to confirm model is saved
    logger.info(f"Model saved: {artifact_path} ✅")

# ==========================
# Function 7: Main entry point
# ==========================

def main():
    """
    Orchestrate full Logistic Regression training pipeline with MLflow tracking.

    WHY this pipeline:
    - Ensures end-to-end automation (no manual steps)
    - Tracks experiments → critical for model comparison & reproducibility
    - Follows production ML lifecycle (data → train → evaluate → deploy-ready)

    Flow:
    Load → Prepare → SMOTE → Train → Evaluate → Save → Log
    """

    logger.info("=" * 60)
    logger.info("Logistic Regression — START")
    logger.info("=" * 60)
    # WHY: Clear visual separation in logs → helps debugging in long pipelines

    start_time = time.time()
    # WHY: Track total pipeline execution time (important for optimization & SLAs)

    # Set MLflow experiment name
    mlflow.set_experiment("aml-fraud-detection")
    # WHY:
    # - Groups all runs under one experiment
    # - Makes it easy to compare multiple model versions (RF vs XGBoost vs tuning)

    with mlflow.start_run(run_name="Logistic Regression"):
        # WHY:
        # - Each run = one experiment execution
        # - Enables tracking parameters, metrics, artifacts in a structured way

        # Step 1: Load Gold data
        df = load_data(GOLD_FILE)
        # WHY:
        # - Gold layer = clean, validated, model-ready data
        # - Avoids reprocessing raw/dirty data every time

        # Step 2: Prepare features and target
        X_train, X_test, y_train, y_test = prepare_data(df)
        # WHY:
        # - Splitting ensures unbiased evaluation
        # - Prevents data leakage (model should not see test data during training)

        # Step 3: Apply SMOTE
        X_train, y_train = apply_smote(X_train, y_train)
        # WHY:
        # - AML datasets are highly imbalanced (very few fraud cases)
        # - SMOTE synthetically generates minority samples
        # - Helps model learn fraud patterns better

        # Step 4: Train model
        model = train_model(X_train, y_train)
        # WHY:
        # - Uses Logistic Regression as baseline model
        # - Simple linear model — fast and interpretable ✅

        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        # WHY:
        # - Evaluation on unseen data = real performance
        # - Prevents overfitting assumptions

        # Step 6: Log parameters to MLflow
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("test_size", TEST_SIZE)
        # WHY:
        # - Parameters must be tracked to reproduce results later
        # - Helps compare which configuration worked best

        # Step 7: Log metrics to MLflow
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("auc_roc", metrics["auc_roc"])
        # WHY:
        # - Metrics define model success
        # - AML focus → recall & AUC are critical (catch fraud cases)
        # - Enables experiment comparison in MLflow UI

        # Step 8: Save model locally
        save_model(model, ARTIFACT_PATH)
        # WHY:
        # - Persist model for inference (API / batch scoring)
        # - Needed for deployment outside MLflow as well

        # Step 9: Log model to MLflow
        mlflow.sklearn.log_model(model, "Logistic Regression")
        # WHY:
        # - Stores model as artifact in MLflow
        # - Enables versioning, registry, and easy deployment

        logger.info("MLflow tracking complete ✅")

    elapsed = time.time() - start_time
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    # WHY:
    # - Helps identify performance bottlenecks (SMOTE / training / I/O)

    logger.info("=" * 60)
    logger.info("Logistic Regression — COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================

if __name__ == "__main__":
    main()
