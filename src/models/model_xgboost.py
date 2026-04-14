"""
model_xgboost.py
-----------------
Purpose : Train and evaluate XGBoost model on Gold layer data.
Layer   : ML Models — Phase 3
Input   : data/gold/aml_gold.parquet
Output  : artifacts/xgboost.pkl
Author  : Dhiru Misra
Version : 1.0.0
"""
# ==========================
# Standard library imports
# ==========================
import os              # Used to dynamically build file paths → avoids hardcoding (works across environments)
import logging         # Needed for tracking execution, debugging, and production monitoring
import time            # Helps measure execution time (useful for performance tuning)
import joblib          # Used to persist (save/load) trained ML models efficiently

# ==========================
# Third party imports
# ==========================
import pandas as pd    # Core library for structured data handling (DataFrames)
import numpy as np     # Efficient numerical computations (used internally by ML models)

from xgboost import XGBClassifier   # Boosting algorithm → better for tabular + imbalanced data (like AML)

from sklearn.model_selection import train_test_split  # To split data into train/test for unbiased evaluation

from sklearn.metrics import (
    precision_score,      # Important for AML → minimizes false positives
    recall_score,         # Critical → ensures fraud cases are not missed
    f1_score,             # Balance between precision & recall
    roc_auc_score,        # Measures ranking ability of model (very important in imbalance)
    confusion_matrix,     # Helps analyze TP, FP, FN, TN
    classification_report # Full summary of model performance
)

from imblearn.over_sampling import SMOTE   # Used because AML data is highly imbalanced (very few fraud cases)

# ==========================
# Logging configuration
# ==========================
logging.basicConfig(
    level=logging.INFO,   # INFO level ensures we log important steps without too much noise (DEBUG is too verbose)
    format="%(asctime)s | %(levelname)s | %(message)s",  # Structured logs → easier to debug in pipelines
    datefmt="%Y-%m-%d %H:%M:%S"  # Standard readable timestamp format
)

logger = logging.getLogger(__name__)  # Create module-level logger (best practice for scalable projects)

# ==========================
# Constants
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
# WHY: Ensures script works regardless of where it's executed from (no relative path issues)

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  
# WHY: Moves 2 levels up → typical project structure (src → project root)

GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "aml_gold.parquet")  
# WHY: Gold layer contains clean, transformed, model-ready data (Medallion architecture best practice)

ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "xgboost.pkl")  
# WHY: Store trained model separately → enables reuse in inference pipeline

TARGET_COLUMN = "is_laundering"  
# WHY: This is the business label → predicting suspicious transactions (core AML objective)

TEST_SIZE = 0.2  
# WHY: 80-20 split is industry standard → enough data for training + reliable validation

RANDOM_STATE = 42  
# WHY: Fixes randomness → ensures reproducibility (same results every run, critical in production & debugging)

N_ESTIMATORS = 200  
# WHY:
# - 200 trees provide enough learning capacity for complex AML patterns
# - Works well with learning_rate=0.1 (balanced config)
# - Avoids underfitting (<100) and overfitting (>500)
# - Keeps training time reasonable
# - RF  → N_ESTIMATORS = 100  (independent trees — 100 is enough)
# - XGB → N_ESTIMATORS = 200  (sequential — more rounds = better correction)

LEARNING_RATE = 0.1  
# WHY:
# - Controls how fast model learns
# - 0.1 is a standard starting point (fast + stable)
# - Lower values (0.01) need more trees → slower training

SCALE_POS_WEIGHT = 14285  
# WHY:
# - Dataset is highly imbalanced (fraud cases are very rare)
# - This value ≈ (negative samples / positive samples)
# - Forces model to pay more attention to minority class (fraud detection)
# - Better alternative to only using SMOTE in many real-world cases

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
    Train XGBoost model on SMOTE balanced data.

    WHY:
    - XGBoost is chosen because it performs well on tabular + imbalanced data
    - Boosting helps sequentially reduce errors (better than bagging for AML use case)

    Args:
        X_train (pd.DataFrame): Balanced training features
        y_train (pd.Series)   : Balanced training labels

    Returns:
        XGBClassifier: Trained model
    """

    logger.info("Training XGBoost model...")
    logger.info(f"Number of estimators : {N_ESTIMATORS}")   # WHY: To track model complexity during runs (useful for tuning)
    logger.info(f"Learning rate        : {LEARNING_RATE}")  # WHY: Helps debug convergence issues if model under/overfits
    logger.info(f"Scale pos weight     : {SCALE_POS_WEIGHT}")  # WHY: Critical for imbalance → must be visible in logs

    model = XGBClassifier(

        n_estimators=N_ESTIMATORS,
        # WHY:
        # - Controls number of boosting rounds (trees)
        # - 200 is chosen as a balance between performance and training time
        # - Too low → underfitting, too high → overfitting + slower training

        learning_rate=LEARNING_RATE,
        # WHY:
        # - Controls how much each tree contributes
        # - 0.1 is standard → fast learning without instability
        # - Works well with 200 trees (balanced combo)

        scale_pos_weight=SCALE_POS_WEIGHT,
        # WHY:
        # - Handles class imbalance internally
        # - Gives higher importance to minority class (fraud cases)
        # - Value is roughly (non-fraud / fraud ratio)
        # - More effective than only using SMOTE in many real scenarios

        random_state=RANDOM_STATE,
        # WHY:
        # - Ensures reproducibility (same model every run)
        # - Critical for debugging, audits, and production consistency

        use_label_encoder=False,
        # WHY:
        # - Disables old internal label encoding (deprecated in newer XGBoost)
        # - Prevents unnecessary warnings

        eval_metric="auc",
        # WHY:
        # - AUC is best metric for imbalanced classification
        # - Measures ranking ability (not just accuracy)
        # - Important for AML → we care about separating fraud vs non-fraud

        n_jobs=-1
        # WHY:
        # - Uses all CPU cores → speeds up training
        # - Important when dataset is large (like AML datasets)
    )

    model.fit(X_train, y_train)
    # WHY:
    # - Trains model on balanced dataset (SMOTE applied before)
    # - Learns patterns of fraud vs non-fraud

    logger.info("Model training complete ✅")

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
    Orchestrate full XGBoost training pipeline.
    Load → Prepare → SMOTE → Train → Evaluate → Save
    """
    logger.info("=" * 60)
    logger.info("XGBoost Pipeline — START")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Load Gold data
    df = load_data(GOLD_FILE)

    # Step 2: Prepare features and target — 80/20 split
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Apply SMOTE on training data only
    X_train, y_train = apply_smote(X_train, y_train)

    # Step 4: Train XGBoost model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model on test data
    metrics = evaluate_model(model, X_test, y_test)

    # Step 6: Save trained model to artifacts
    save_model(model, ARTIFACT_PATH)

    elapsed = time.time() - start_time
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    logger.info("=" * 60)
    logger.info("XGBoost Pipeline — COMPLETE")
    logger.info("=" * 60)

# ==========================
# Script entry point
# ==========================
if __name__ == "__main__":
    main()