"""Integration coverage for the README five-minute demo path."""

import json
import subprocess
import sys
from pathlib import Path


def test_hello_world_generates_sample_report_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tracked_reports = [
        repo_root / "examples" / "reports" / "hello_world_report.json",
        repo_root / "examples" / "reports" / "hello_world_report.md",
    ]
    original_reports = {path: path.read_bytes() for path in tracked_reports}

    result = subprocess.run(
        [sys.executable, "examples/hello_world.py", "--output-dir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report_json = tmp_path / "hello_world_report.json"
    report_md = tmp_path / "hello_world_report.md"

    assert "tracelens hello-world" in result.stdout
    assert result.stdout.count("add-2-2") == 1
    assert "runs=3" in result.stdout
    assert report_json.exists()
    assert report_md.exists()

    data = json.loads(report_json.read_text())
    assert data["total_tasks"] == 3
    assert data["total_trials"] == 9
    assert data["pass_at_k"]["pass@3"] == 1.0
    assert data["reliability"]["pass^3"] == 1.0

    markdown = report_md.read_text()
    assert "## Graders" in markdown
    assert "## Baseline Comparison" in markdown
    assert "## Regression Result" in markdown
    assert "## CI Summary" in markdown
    assert {path: path.read_bytes() for path in tracked_reports} == original_reports
