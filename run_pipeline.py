"""Deterministic pipeline WITHOUT LLM agents (no API key needed).

Use it to test the data/ML logic quickly, or to regenerate artifacts for the app.
The full multi-agent version is `python main.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn import analyst_core as A  # noqa: E402
from churn import scientist_core as S  # noqa: E402
from churn.logging_utils import get_logger  # noqa: E402
from churn.validation import ValidationError  # noqa: E402

log = get_logger("pipeline")


def main() -> int:
    try:
        log.info("=== Crew 1 steps (deterministic) ===")
        A.ingest_and_clean()
        A.run_eda()
        A.build_contract()
        A.write_insights_template()
        log.info("=== Handoff gate ===")
        S.validate_inputs()
        log.info("=== Crew 2 steps (deterministic) ===")
        S.build_features()
        S.train_and_compare()
        S.write_model_card_template()
        log.info("Pipeline finished OK")
        return 0
    except ValidationError as e:
        log.error("Pipeline stopped at gate '%s': %s", e.gate, e.errors)
        return 1


if __name__ == "__main__":
    sys.exit(main())
