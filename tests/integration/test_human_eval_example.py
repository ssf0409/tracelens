"""Integration coverage for the human-eval calibration example."""

import subprocess
import sys
from pathlib import Path


def test_human_eval_example_reports_drift_and_disagreements() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "examples/human_eval_calibration.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    out = result.stdout

    # Runs without any API key and produces a calibration report.
    assert "Calibration Report" in out
    # The recorded grader is deliberately miscalibrated, so drift is detected.
    assert "DRIFT DETECTED" in out
    # Disagreement cases are surfaced (grader passed something a human failed).
    assert "Disagreements" in out
    assert "t2" in out
    # And a concrete next step is recommended.
    assert "Recommended action" in out
