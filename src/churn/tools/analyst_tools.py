"""CrewAI tools for the Data Analyst Crew. Each tool runs deterministic code and
returns a JSON summary the agent can reason about."""
import json

from crewai.tools import tool

from .. import analyst_core as A


def _safe(fn):
    try:
        return json.dumps(fn(), default=str)
    except Exception as e:  # report the error to the agent instead of crashing the crew
        return json.dumps({"status": "error", "error": str(e)})


@tool("ingest_and_clean_dataset")
def ingest_and_clean_tool() -> str:
    """Load the raw Telco churn CSV, validate its columns, clean it (types, blanks,
    duplicates) and save artifacts/analyst/clean_data.csv. Returns a JSON summary."""
    return _safe(A.ingest_and_clean)


@tool("run_exploratory_analysis")
def run_eda_tool() -> str:
    """Run descriptive analytics and EDA on clean_data.csv. Saves eda_report.html (with
    charts) and eda_stats.json. Returns the statistics as JSON."""
    return _safe(A.run_eda)


@tool("build_dataset_contract")
def build_contract_tool() -> str:
    """Generate dataset_contract.json (schema, allowed values, assumptions, constraints)
    from clean_data.csv. Returns a JSON summary."""
    return _safe(A.build_contract)


@tool("read_eda_statistics")
def read_eda_stats_tool() -> str:
    """Return the EDA statistics (churn rates by segment, revenue lost) as JSON."""
    return _safe(A.read_eda_stats)
