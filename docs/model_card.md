# Model Card — AML Transaction Monitoring

## Model Details

| Item | Detail |
|---|---|
| Model Type | XGBoost Classifier |
| Version | 1.0.0 |
| Author | Dhiru Misra |
| Date | April 2026 |
| Framework | XGBoost + Scikit-learn |

---

## Problem Statement

Detect money laundering transactions from 179 million banking records. Extreme class imbalance — 1 suspicious transaction per 14,285 legitimate transactions.

---

## Training Data

| Item | Detail |
|---|---|
| Dataset | IBM AML Synthetic Dataset |
| Total rows | 179,702,225 |
| DEV sample | 500,000 rows |
| Features | 16 engineered features |
| Target | is_laundering (0/1) |
| Class ratio | 14,285:1 (legitimate:suspicious) |

---

## Model Comparison

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.0008 | 0.5714 | 0.0016 | 0.9322 |
| Random Forest | 0.1154 | 0.4286 | 0.1818 | 0.8544 |
| **XGBoost** | **0.0256** | **0.4286** | **0.0484** | **0.9877** |

**Selected model: XGBoost — highest AUC-ROC**

---

## XGBoost Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| n_estimators | 200 | Balance accuracy vs speed |
| learning_rate | 0.1 | Standard stable starting point |
| scale_pos_weight | 14285 | Handles class imbalance |
| eval_metric | auc | Best metric for imbalanced data |

---

## Key Features (SHAP)

1. payment_format — most important fraud signal
2. to_bank — destination bank identity
3. amount_received — transaction amount
4. hour_of_day — time of transaction
5. from_bank — source bank identity

---

## Limitations

- Trained on synthetic data — real world performance may differ
- DEV environment uses 500K sample — full 179M in production
- Low precision expected due to extreme class imbalance
- Model requires retraining as fraud patterns evolve

---

## Intended Use

- Internal AML compliance teams
- Flagging suspicious transactions for human review
- Generating Suspicious Activity Reports (SARs)
- NOT for automated account blocking without human review