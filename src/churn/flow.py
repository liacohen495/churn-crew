"""CrewAI Flow: Data Analyst Crew -> validation gate -> Data Scientist Crew -> validation gate.

Every step logs to logs/pipeline.log. Any failure is caught, written to
artifacts/run_status.json, and the flow stops cleanly instead of crashing.
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from . import config as C
from .crews.analyst_crew import build_analyst_crew
from .crews.scientist_crew import build_scientist_crew
from .logging_utils import get_logger
from .scientist_core import CORE_FEATURES, FEATURE_LIST
from .validation import (
    check_contract, check_files_exist, check_required_features, load_contract,
)

log = get_logger("flow")

ANALYST_OUTPUTS = [C.CLEAN_DATA, C.EDA_REPORT, C.INSIGHTS, C.CONTRACT]
SCIENTIST_OUTPUTS = [C.FEATURES, C.MODEL, C.EVAL_REPORT, C.MODEL_CARD, C.SCORED]


class ChurnState(BaseModel):
    simulate_bad_data: bool = False
    stage: str = "not_started"
    errors: list[str] = Field(default_factory=list)
    selected_model: str | None = None


class ChurnFlow(Flow[ChurnState]):

    # ------------------------------------------------------------------ 0. preflight
    @start()
    def preflight(self):
        self.state.stage = "preflight"
        log.info("=== Flow started ===")
        if not C.DATA_RAW.exists():
            self.state.errors.append(f"Raw dataset missing: {C.DATA_RAW}")
        model = os.getenv("MODEL", "openai/gpt-4o-mini")
        key_var = {"openai/": "OPENAI_API_KEY", "gemini/": "GEMINI_API_KEY",
                   "groq/": "GROQ_API_KEY"}.get(model.split("/")[0] + "/")
        if key_var and not os.getenv(key_var):
            self.state.errors.append(f"{key_var} is not set (add it to .env) for MODEL={model}")
        # Remove outputs of previous runs so the gates check THIS run's work
        if not self.state.errors:
            for p in ANALYST_OUTPUTS + SCIENTIST_OUTPUTS:
                p.unlink(missing_ok=True)

    @router(preflight)
    def preflight_gate(self):
        return "preflight_failed" if self.state.errors else "preflight_ok"

    # ------------------------------------------------------------------ 1. analyst crew
    @listen("preflight_ok")
    def run_analyst_crew(self):
        self.state.stage = "analyst_crew"
        log.info("=== Crew 1: Data Analyst Crew ===")
        try:
            build_analyst_crew().kickoff()
        except Exception as e:
            self.state.errors.append(f"Analyst crew crashed: {e}")
        if self.state.simulate_bad_data and C.CLEAN_DATA.exists():
            log.warning("SIMULATION: corrupting clean_data.csv to demonstrate the gate")
            df = pd.read_csv(C.CLEAN_DATA).drop(columns=["Contract"])
            df.loc[0, "InternetService"] = "Satellite"
            df.to_csv(C.CLEAN_DATA, index=False)

    @router(run_analyst_crew)
    def handoff_gate(self):
        """Gate 1: artifacts exist AND dataset_contract matches clean_data."""
        self.state.stage = "handoff_gate"
        errs = check_files_exist(ANALYST_OUTPUTS)
        if not errs:
            errs = check_contract(pd.read_csv(C.CLEAN_DATA), load_contract(C.CONTRACT))
        self.state.errors += errs
        for e in errs:
            log.error("[handoff_gate] %s", e)
        if self.state.errors:
            return "handoff_failed"
        log.info("[handoff_gate] passed -> handing off to Data Scientist Crew")
        return "handoff_ok"

    # ------------------------------------------------------------------ 2. scientist crew
    @listen("handoff_ok")
    def run_scientist_crew(self):
        self.state.stage = "scientist_crew"
        log.info("=== Crew 2: Data Scientist Crew ===")
        try:
            build_scientist_crew().kickoff()
        except Exception as e:
            self.state.errors.append(f"Scientist crew crashed: {e}")

    @router(run_scientist_crew)
    def model_gate(self):
        """Gate 2: features complete and every Crew 2 artifact saved."""
        self.state.stage = "model_gate"
        errs = check_files_exist(SCIENTIST_OUTPUTS)
        if C.FEATURES.exists() and FEATURE_LIST.exists():
            listed = json.loads(FEATURE_LIST.read_text())["features"]
            errs += check_required_features(pd.read_csv(C.FEATURES),
                                            sorted(set(listed + CORE_FEATURES)))
        self.state.errors += errs
        for e in errs:
            log.error("[model_gate] %s", e)
        if self.state.errors:
            return "model_failed"
        self.state.selected_model = json.loads(C.METRICS.read_text())["selected_model"]
        return "model_ok"

    # ------------------------------------------------------------------ 3. finish
    @listen("model_ok")
    def finish(self):
        self.state.stage = "done"
        self._write_status("success")
        log.info("=== Flow finished OK — selected model: %s ===", self.state.selected_model)
        return "success"

    @listen(or_("preflight_failed", "handoff_failed", "model_failed"))
    def fail_gracefully(self):
        self._write_status("failed")
        log.error("=== Flow stopped at stage '%s' — %d problem(s). See logs/pipeline.log ===",
                  self.state.stage, len(self.state.errors))
        return "failed"

    def _write_status(self, status: str):
        C.RUN_STATUS.write_text(json.dumps({
            "status": status,
            "stage": self.state.stage,
            "errors": self.state.errors,
            "selected_model": self.state.selected_model,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
