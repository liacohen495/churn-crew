"""CrewAI tools for the Data Scientist Crew."""
import json

from crewai.tools import tool

from .. import scientist_core as S
from ..validation import ValidationError


def _safe(fn):
    try:
        return json.dumps(fn(), default=str)
    except ValidationError as e:
        return json.dumps({"status": "validation_failed", "gate": e.gate, "errors": e.errors})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool("validate_clean_data_against_contract")
def validate_inputs_tool() -> str:
    """Check that clean_data.csv matches dataset_contract.json (columns, types, allowed
    values, nulls, primary key). Returns 'passed' or the list of violations."""
    return _safe(S.validate_inputs)


@tool("engineer_features")
def build_features_tool() -> str:
    """Create the modeling feature table (engineered + one-hot features) and save
    artifacts/scientist/features.csv. Returns a JSON summary."""
    return _safe(S.build_features)


@tool("train_and_compare_models")
def train_and_compare_tool() -> str:
    """Confirm required features exist, train a baseline, logistic regression and random
    forest, compare them on a held-out test set, save the best model (model.joblib),
    evaluation_report.md and scored_customers.csv. Returns metrics as JSON."""
    return _safe(S.train_and_compare)


@tool("read_model_metrics")
def read_metrics_tool() -> str:
    """Return the evaluation metrics, selected model and top churn drivers as JSON."""
    return _safe(S.read_metrics)
