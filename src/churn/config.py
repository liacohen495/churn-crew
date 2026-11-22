"""Central configuration: paths, seeds and constants used by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
ANALYST_DIR = ROOT / "artifacts" / "analyst"
SCIENTIST_DIR = ROOT / "artifacts" / "scientist"
LOG_DIR = ROOT / "logs"

# Crew 1 outputs
CLEAN_DATA = ANALYST_DIR / "clean_data.csv"
EDA_REPORT = ANALYST_DIR / "eda_report.html"
EDA_STATS = ANALYST_DIR / "eda_stats.json"
INSIGHTS = ANALYST_DIR / "insights.md"
CONTRACT = ANALYST_DIR / "dataset_contract.json"

# Crew 2 outputs
FEATURES = SCIENTIST_DIR / "features.csv"
MODEL = SCIENTIST_DIR / "model.joblib"
METRICS = SCIENTIST_DIR / "metrics.json"
EVAL_REPORT = SCIENTIST_DIR / "evaluation_report.md"
MODEL_CARD = SCIENTIST_DIR / "model_card.md"
SCORED = SCIENTIST_DIR / "scored_customers.csv"

RUN_STATUS = ROOT / "artifacts" / "run_status.json"

SEED = 42
TEST_SIZE = 0.2
TARGET = "Churn"
ID_COL = "customerID"

# Risk bands used by the app (probability of churn)
HIGH_RISK = 0.75
MEDIUM_RISK = 0.45

for d in (ANALYST_DIR, SCIENTIST_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
