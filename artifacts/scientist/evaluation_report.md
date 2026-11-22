# Evaluation Report

**Selected model:** `random_forest` (highest ROC-AUC on the held-out test set).
Train / test split: 5,634 / 1,409 rows, stratified, seed 42.
Test-set churn rate: 26.5%. Decision threshold: 0.5.

## Model comparison (held-out test set)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy | Churners in top 10% |
|---|---|---|---|---|---|---|
| baseline_majority | 0.500 | 0.000 | 0.000 | 0.000 | 0.735 | 11.8% |
| logistic_regression | 0.843 | 0.789 | 0.508 | 0.618 | 0.741 | 28.9% |
| random_forest | 0.846 | 0.773 | 0.531 | 0.630 | 0.759 | 28.9% |

**Why not accuracy?** The majority-class baseline already scores
73.5% accuracy by predicting "no churn" for
everyone, and it catches zero churners. ROC-AUC and recall measure what the business needs:
finding the customers who are about to leave.

## Selected model — confusion matrix (threshold 0.5)

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 780 | 255 |
| **Actually churned** | 85 | 289 |

The model catches **289 of 374 churners (77%)**. The cost is 255
false alarms, customers who get a retention offer they didn't need.

## Strongest drivers (logistic regression coefficients)
- Push churn **up**: `TotalCharges`, `Contract_Month-to-month`, `InternetService_Fiber optic`, `is_new_customer`, `PaperlessBilling`
- Push churn **down**: `tenure`, `Contract_Two year`, `MonthlyCharges`, `avg_monthly_spend`, `InternetService_DSL`
