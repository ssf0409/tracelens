"""Integration coverage for the README five-minute demo path.

The example writes its reports to ``examples/reports/`` by default; that is
the checked-in sample the README links to, regenerated only on purpose. The
test therefore runs the example with ``--reports-dir`` pointing at a
temporary directory and checks that the sample files are left untouched.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "hello_world.py"
SAMPLE_DIR = REPO_ROOT / "examples" / "reports"
SAMPLE_FILES = ("hello_world_report.json", "hello_world_report.md")


def test_hello_world_generates_sample_report_artifacts(tmp_path: Path) -> None:
    checked_in = {name: (SAMPLE_DIR / name).read_bytes() for name in SAMPLE_FILES}

    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--reports-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_json = tmp_path / "hello_world_report.json"
    report_md = tmp_path / "hello_world_report.md"

    assert "tracelens hello-world" in result.stdout
    assert result.stdout.count("add-2-2") == 1
    assert "runs=3" in result.stdout
    assert f"report json: {report_json}" in result.stdout
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

    # The checked-in sample reports are illustrative data: a test run never rewrites them.
    for name, before in checked_in.items():
        assert (SAMPLE_DIR / name).read_bytes() == before, name


def test_hello_world_defaults_to_the_checked_in_reports_directory() -> None:
    spec = importlib.util.spec_from_file_location("hello_world_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_REPORTS_DIR == SAMPLE_DIR
    assert module._parse_args([]).reports_dir == SAMPLE_DIR
    assert module._parse_args(["--reports-dir", "/tmp/elsewhere"]).reports_dir == Path("/tmp/elsewhere")
