"""Crew 1 — Data Analyst Crew (4 agents).

Question it answers: "What has happened in the business?"
Outputs: clean_data.csv, eda_report.html, insights.md, dataset_contract.json
"""
import os

from crewai import Agent, Crew, Process, Task

from ..config import INSIGHTS
from ..llm import get_llm
from ..tools.analyst_tools import (
    build_contract_tool, ingest_and_clean_tool, read_eda_stats_tool, run_eda_tool,
)


def build_analyst_crew() -> Crew:
    llm = get_llm()

    data_engineer = Agent(
        role="Data Engineer",
        goal="Ingest the raw customer dataset, validate it and produce a clean, documented file.",
        backstory="You own data quality at a retail-tech company. You never guess: you run "
                  "the cleaning tool and report exactly what it changed.",
        tools=[ingest_and_clean_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    eda_analyst = Agent(
        role="EDA Analyst",
        goal="Describe what has happened: churn levels and which customer segments churn most.",
        backstory="You are a descriptive-analytics specialist. All numbers you report come "
                  "from the EDA tool, never from memory.",
        tools=[run_eda_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    contract_owner = Agent(
        role="Data Contract Owner",
        goal="Publish a dataset contract the Data Scientist crew must follow.",
        backstory="You define schemas, allowed values, assumptions and constraints so that "
                  "downstream teams can trust the data.",
        tools=[build_contract_tool], llm=llm, verbose=True, allow_delegation=False,
    )
    insights_writer = Agent(
        role="Business Insights Writer",
        goal="Turn EDA statistics into a short, actionable business summary for management.",
        backstory="You translate data into decisions for non-technical managers. You only use "
                  "numbers returned by the statistics tool.",
        tools=[read_eda_stats_tool], llm=llm, verbose=True, allow_delegation=False,
    )

    clean = Task(
        description="Run the `ingest_and_clean_dataset` tool once. Report rows, columns, "
                    "churn rate and every cleaning note it returns.",
        expected_output="A short data-quality summary with the exact figures from the tool.",
        agent=data_engineer,
    )
    eda = Task(
        description="Run the `run_exploratory_analysis` tool once. Summarise the overall churn "
                    "rate and the 3 segments with the highest churn rate.",
        expected_output="Bullet list of the main descriptive findings with exact percentages.",
        agent=eda_analyst, context=[clean],
    )
    contract = Task(
        description="Run the `build_dataset_contract` tool once. Summarise what the contract "
                    "guarantees to the Data Scientist crew.",
        expected_output="A summary of the contract: number of columns, target, key constraints.",
        agent=contract_owner, context=[clean],
    )
    insights = Task(
        description=(
            "Call `read_eda_statistics`. Write insights.md in Markdown for a retention manager:\n"
            "1. A one-sentence headline with the churn rate and monthly revenue lost.\n"
            "2. '## Key findings' — 5 numbered findings, each with exact percentages from the tool.\n"
            "3. '## Recommendations' — 4 concrete retention actions linked to the findings.\n"
            "Use only numbers from the tool. Output only the Markdown."
        ),
        expected_output="A Markdown business-insights document.",
        agent=insights_writer, context=[eda], output_file=str(INSIGHTS),
    )

    return Crew(
        agents=[data_engineer, eda_analyst, contract_owner, insights_writer],
        tasks=[clean, eda, contract, insights],
        process=Process.sequential,
        verbose=True,
        max_rpm=int(os.getenv("MAX_RPM", "8")),  # stays inside free-tier rate limits
    )
