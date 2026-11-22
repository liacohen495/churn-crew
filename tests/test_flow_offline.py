"""Offline tests: run the Flow routing and gates with the crews replaced by their
deterministic tools (no API key, no LLM). Run with:  python -m pytest -q
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import analyst_core as A  # noqa: E402
from churn import config as C  # noqa: E402
from churn import flow as F  # noqa: E402
from churn import scientist_core as S  # noqa: E402


class _FakeCrew:
    def __init__(self, steps):
        self.steps = steps

    def kickoff(self):
        for s in self.steps:
            s()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(F, "build_analyst_crew", lambda: _FakeCrew(
        [A.ingest_and_clean, A.run_eda, A.build_contract, A.write_insights_template]))
    monkeypatch.setattr(F, "build_scientist_crew", lambda: _FakeCrew(
        [S.validate_inputs, S.build_features, S.train_and_compare, S.write_model_card_template]))


def _status():
    return json.loads(C.RUN_STATUS.read_text())


def test_flow_fails_gracefully_on_bad_data():
    F.ChurnFlow().kickoff(inputs={"simulate_bad_data": True})
    st = _status()
    assert st["status"] == "failed"
    assert st["stage"] == "handoff_gate"
    assert any("Contract" in e for e in st["errors"])
    assert not C.MODEL.exists()  # Crew 2 never ran


def test_flow_succeeds():
    F.ChurnFlow().kickoff(inputs={"simulate_bad_data": False})
    st = _status()
    assert st["status"] == "success", st
    assert st["selected_model"] in {"logistic_regression", "random_forest"}
    for p in F.ANALYST_OUTPUTS + F.SCIENTIST_OUTPUTS:
        assert p.exists(), p


def test_metrics_are_deterministic():
    # runs after test_flow_succeeds, so features.csv exists
    S.train_and_compare()
    first = json.loads(C.METRICS.read_text())
    S.train_and_compare()
    assert json.loads(C.METRICS.read_text()) == first
