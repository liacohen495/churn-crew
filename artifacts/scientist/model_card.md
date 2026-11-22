# Model Card — Customer Churn Risk

## Model purpose
Rank telecom customers by their probability of leaving in the next month, so the retention
team can contact the riskiest customers first. Intended user: a retention / customer-success manager.
Not intended for pricing, credit or any decision that denies a customer service.

## Model
`random_forest` chosen over logistic regression / random forest by ROC-AUC.
Per-customer reasons come from the logistic regression variant (coefficient × scaled value).

## Training data summary
Kaggle "Telco Customer Churn" (IBM sample), 7,043 customers, 20 attributes plus engineered features.
5,634 rows used for training, 1,409 held out. Churn rate ≈ 26.5%.

## Metrics (held-out test set)
- ROC-AUC: **0.846**
- Recall: **77.3%** — share of churners the model flags
- Precision: 53.1%
- Accuracy: 75.9% (majority baseline: 73.5%)
- Calling the riskiest 10% of customers reaches **29%** of all churners

## Limitations
- Single snapshot from one (fictional) US telecom; patterns may not transfer to other markets or time periods.
- No time dimension: the model cannot tell when a customer will leave.
- Scores for training-set customers are optimistic; only the test set measures true performance.
- Reasons are associations, not causes.

## Ethical considerations
- `gender` and `SeniorCitizen` are in the data. They are used as inputs here; a production
  version should test whether removing them changes fairness or accuracy.
- Flagging a customer must lead to a better offer, never to worse service.
- Customer IDs are pseudonymous; no personal contact data is used.
