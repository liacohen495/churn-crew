"""Run the full multi-agent CrewAI Flow.

    python main.py                      # normal run
    python main.py --simulate-bad-data  # demo: break the data and watch the gate stop the flow
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()

from churn.config import RUN_STATUS  # noqa: E402
from churn.flow import ChurnFlow  # noqa: E402


def main() -> int:
    flow = ChurnFlow()
    flow.kickoff(inputs={"simulate_bad_data": "--simulate-bad-data" in sys.argv})
    status = json.loads(RUN_STATUS.read_text())
    print("\nRUN STATUS:", json.dumps(status, indent=2))
    return 0 if status["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
