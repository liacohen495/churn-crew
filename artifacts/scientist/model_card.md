# Model Card — Customer Churn Risk

## Model purpose (who uses it and for what decision; what it must NOT be used for)
* **Intended Users:** Customer Retention, Customer Success, and Marketing teams.
* **Intended Decision:** Identifying customers who are at high risk of churning so that retention teams can proactively deploy targeted customer support, personalized incentives, and service enhancements.
* **Prohibited Uses:** This model must **NOT** be used for punitive decisions, service degradation, automatic contract termination, discriminatory pricing, credit scoring, or denying standard customer service or support.

## Training data summary (Kaggle Telco Customer Churn, rows, split, churn rate)
* **Data Source:** Kaggle Telco Customer Churn dataset.
* **Training Set:** 5,634 rows (80% of total dataset).
* **Test Set:** 1,409 rows (20% of total dataset).
* **Total Rows:** 7,043 rows.
* **Test Set Churn Rate:** 0.2654 (26.54%).

## Metrics (selected model vs the other variants and the baseline, exact numbers)
**Selected Model:** Random Forest (Selected based on highest ROC-AUC on the held-out test set at a decision threshold of 0.5).

### Evaluation Comparison Table

| Metric | Baseline Majority | Logistic Regression | Random Forest (Selected) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.5000 | 0.8426 | **0.8458** |
| **Recall** | 0.0000 | **0.7888** | 0.7727 |
| **Accuracy** | 0.7346 | 0.7410 | **0.7587** |
| **Precision** | 0.0000 | 0.5077 | **0.5312** |
| **F1-Score** | 0.0000 | 0.6178 | **0.6296** |
| **Top 10% Capture** | 0.1176 | 0.2888 | **0.2888** |

### Selected Model Confusion Matrix (Random Forest, N=1,409)
* **True Negatives (TN):** 780
* **False Positives (FP):** 255
* **False Negatives (FN):** 85
* **True Positives (TP):** 289

## Limitations
* **Precision & False Positives:** At a decision threshold of 0.5, Random Forest achieves a precision of 0.5312, resulting in 255 false positives out of 1,409 test samples. Outgoing retention offers may be directed to non-churning customers.
* **False Negatives:** With a recall of 0.7727, the selected model misses 85 actual churners (22.73% of churners in the test set).
* **Data Drift & Static Scope:** The model is trained on a single static snapshot of telecom data. Changes in market competition, pricing structures, or customer behavior over time will require periodic model retraining and re-evaluation.

## Ethical considerations (gender and SeniorCitizen are inputs; how flags are used)
* **Demographic Input Features:** The dataset includes demographic variables such as `gender` and `SeniorCitizen`.
* **Governance and Fair Treatment:** Churn risk predictions must be monitored to prevent disproportionate impact or discriminatory treatment across demographic sub-groups.
* **Usage Restrictions:** Risk flags derived from models containing demographic features must only be used for positive intervention (e.g., offering loyalty discounts, enhanced customer service) and never for adverse actions, pricing surcharges, or reduced support access.