"""``scripts/compare_error_rates.py`` reproduces the contract's compare tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_error_rates.py"
spec = importlib.util.spec_from_file_location("compare_error_rates", SCRIPT)
assert spec is not None and spec.loader is not None
compare_error_rates = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare_error_rates  # dataclasses resolve annotations through it
spec.loader.exec_module(compare_error_rates)

REGRESSION_TABLE = "**Error rates.** How often the verdict is a regression"
EXIT_ZERO_TABLE = "**Exit 0.** How often the verdict exits 0"
REGENERATE = (
    "docs/statistical-contract.md no longer quotes what the script computes: rerun "
    "`uv run --no-sync python scripts/compare_error_rates.py` and paste its output"
)


def _contract_row(table: str, tasks: int) -> list[str]:
    """The row for ``tasks`` of the contract table that starts with ``table``, cell by cell."""
    text = (ROOT / "docs" / "statistical-contract.md").read_text(encoding="utf-8")
    after = text.split(table, 1)[1]
    for line in after.splitlines():
        if line.startswith(f"| {tasks} | "):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]
    raise AssertionError(f"no row for {tasks} tasks in the contract table {table!r}")


@pytest.mark.parametrize("tasks", [2, 6])
def test_the_contract_quotes_what_the_script_computes(tasks: int) -> None:
    rates = compare_error_rates.simulate("no change", tasks)
    row = _contract_row(REGRESSION_TABLE, tasks)
    pct = compare_error_rates.pct
    assert row[1:3] == [pct(rates.interval_alone), pct(rates.regression)], REGENERATE


def test_the_contract_quotes_how_often_a_regression_of_the_threshold_exits_0() -> None:
    rates = compare_error_rates.simulate("drop of the threshold, sd the threshold", 6)
    row = _contract_row(EXIT_ZERO_TABLE, 6)
    assert row[1] == compare_error_rates.pct(rates.exit_zero), REGENERATE


def test_no_verdict_below_six_tasks_whatever_the_scenario() -> None:
    quick = compare_error_rates.Settings(runs=50, n_bootstrap=200)
    for scenario in compare_error_rates.SCENARIOS:
        rates = compare_error_rates.simulate(scenario, 5, quick)
        assert rates.regression == rates.exit_zero == 0.0


def test_main_prints_both_tables(capsys: pytest.CaptureFixture[str]) -> None:
    assert compare_error_rates.main(["--runs", "3", "--bootstrap", "100"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(REGRESSION_TABLE)
    assert "(threshold 0.03, confidence 0.95, B = 100;" in out
    assert "| tasks T | no change, interval alone | no change | drop of 0.10 |" in out
    assert EXIT_ZERO_TABLE in out and "| tasks T | drop of τ | no change |" in out
    assert out.count("| 30 | ") == 2


def test_every_setting_can_be_overridden(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--tasks", "6,7", "--runs", "2", "--bootstrap", "50", "--threshold", "0.05",
        "--confidence", "0.9", "--sd", "0.2", "--trials", "10",
    ]
    assert compare_error_rates.main(argv) == 0
    out = capsys.readouterr().out
    assert "(threshold 0.05, confidence 0.9, B = 50;" in out
    assert "standard deviation 0.2" in out and "10 times a side" in out
    assert "Below 5 tasks there is no verdict" in out
    assert "whose standard deviation is the threshold, 0.05: around -0.05" in out
    assert out.count("| 7 | ") == 2 and "| 30 | " not in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--runs", "0"],
        ["--bootstrap", "0"],
        ["--tasks", "6,0"],
        ["--tasks", "six"],
        ["--threshold", "0"],
        ["--sd", "-0.1"],
        ["--trials", "0"],
        ["--confidence", "1"],
        ["--confidence", "0.9999999999996"],
    ],
)
def test_a_setting_that_cannot_run_is_a_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        compare_error_rates.main(argv)
    assert exc.value.code == 2
