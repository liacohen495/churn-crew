"""Crew 2 — Data Scientist Crew (4 agents).

Question it answers: "What is likely to happen next?"
Outputs: features.csv, model.joblib, evaluation_report.md, model_card.md
"""
import os

from crewai import Agent, Crew, Process, Task

from ..config import MODEL_CARD
from ..llm import get_llm
from ..tools.scientist_tools import (
    build_features_tool, read_metrics_tool, train_and_compare_tool, validate_inputs_tool,
)


def build_scientist_crew() -> Crew:
    llm = get_llm()

    validator = Agent(
        role="Data Validator",
        goal="Confirm the cleaned dataset honours the dataset contract before any modeling.",
        backstory="You are the gatekeeper between analytics and data science. If the contract "
                  "is violated you say so clearly and list every violation.",
        tools=[validate_inputs_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    feature_engineer = Agent(
        role="Feature Engineer",
        goal="Build a numeric feature table that captures customer behaviour.",
        backstory="You turn raw customer attributes into model-ready signals.",
        tools=[build_features_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    ml_engineer = Agent(
        role="ML Engineer",
        goal="Train and compare model variations and pick the best one for finding churners.",
        backstory="You care about recall and ROC-AUC, not just accuracy, because the business "
                  "wants to catch customers before they leave.",
        tools=[train_and_compare_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    card_writer = Agent(
        role="Model Card Writer",
        goal="Document the model honestly for business and compliance readers.",
        backstory="You write model cards following responsible-AI practice. You only use "
                  "numbers from the metrics tool.",
        tools=[read_metrics_tool], llm=llm, verbose=True, allow_delegation=False,
    )

    validate = Task(
        description="Run `validate_clean_data_against_contract` once and report the result. "
                    "If it did not pass, list every violation.",
        expected_output="'passed' with row count, or the list of contract violations.",
        agent=validator,
    )
    features = Task(
        description="Run `engineer_features` once. Report the number of features and list the "
                    "engineered features with a one-line business meaning for each.",
        expected_output="Feature summary.",
        agent=feature_engineer, context=[validate],
    )
    train = Task(
        description="Run `train_and_compare_models` once. Compare the models on ROC-AUC, recall "
                    "and accuracy, and explain why the selected model was chosen.",
        expected_output="Model comparison summary with exact metrics.",
        agent=ml_engineer, context=[features],
    )
    card = Task(
        description=(
            "Call `read_model_metrics`. Write model_card.md in Markdown with these sections:\n"
            "# Model Card — Customer Churn Risk\n"
            "## Model purpose (who uses it and for what decision; what it must NOT be used for)\n"
            "## Training data summary (Kaggle Telco Customer Churn, rows, split, churn rate)\n"
            "## Metrics (selected model vs the other variants and the baseline, exact numbers)\n"
            "## Limitations\n"
            "## Ethical considerations (gender and SeniorCitizen are inputs; how flags are used)\n"
            "Use only numbers from the tool. Output only the Markdown."
        ),
        expected_output="A complete Markdown model card.",
        agent=card_writer, context=[train], output_file=str(MODEL_CARD),
    )

    return Crew(
        agents=[validator, feature_engineer, ml_engineer, card_writer],
        tasks=[validate, features, train, card],
        process=Process.sequential,
        verbose=True,
        max_rpm=int(os.getenv("MAX_RPM", "8")),  # stays inside free-tier rate limits
    )
