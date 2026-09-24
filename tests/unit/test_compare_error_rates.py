"""``scripts/compare_error_rates.py`` reproduces the contract's compare table."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_error_rates.py"
spec = importlib.util.spec_from_file_location("compare_error_rates", SCRIPT)
assert spec is not None and spec.loader is not None
compare_error_rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_error_rates)


def _contract_row(tasks: int) -> list[str]:
    """The contract's compare error-rate row for ``tasks``, cell by cell."""
    text = (ROOT / "docs" / "statistical-contract.md").read_text(encoding="utf-8")
    table = text.split("**Error rates.** How often the verdict is a regression", 1)[1]
    for line in table.splitlines():
        if line.startswith(f"| {tasks} | "):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]
    raise AssertionError(f"no row for {tasks} tasks in the contract's compare table")


@pytest.mark.parametrize("tasks", [2, 6])
def test_the_contract_quotes_what_the_script_computes(tasks: int) -> None:
    verdict, interval = compare_error_rates.regression_rates("no change", tasks)
    row = _contract_row(tasks)
    assert row[1] == compare_error_rates.pct(interval)
    assert row[2] == compare_error_rates.pct(verdict)


def test_no_verdict_below_six_tasks_whatever_the_scenario() -> None:
    for scenario in compare_error_rates.SCENARIOS:
        verdict, _ = compare_error_rates.regression_rates(
            scenario, 5, runs=50, n_bootstrap=200
        )
        assert verdict == 0.0


def test_main_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert compare_error_rates.main(["--runs", "3", "--bootstrap", "100"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("**Error rates.** How often the verdict is a regression")
    assert "| tasks T | no change, interval alone | no change | drop of 0.10 |" in out
    assert "| 30 | " in out
