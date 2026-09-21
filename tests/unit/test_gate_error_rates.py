"""``scripts/gate_error_rates.py`` reproduces the contract's error-rate tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gate_error_rates.py"
spec = importlib.util.spec_from_file_location("gate_error_rates", SCRIPT)
assert spec is not None and spec.loader is not None
gate_error_rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_error_rates)


def test_power_and_false_alarm_numbers_quoted_in_the_contract() -> None:
    block = gate_error_rates.block_probability
    # Power for 1.0 -> 0.4, one task (T=1, level 0.05).
    assert block(5, 5, 1.0, 0.4, 0.05) == pytest.approx(0.683, abs=0.002)
    assert block(10, 10, 1.0, 0.4, 0.05) == pytest.approx(0.945, abs=0.002)
    # ... and under Holm over fifty tasks (level 0.001).
    assert block(5, 5, 1.0, 0.4, 0.001) == pytest.approx(0.078, abs=0.002)
    assert block(20, 20, 1.0, 0.4, 0.001) == pytest.approx(0.979, abs=0.002)
    # False alarm on an unchanged flaky task, uncorrected and corrected.
    assert block(5, 5, 0.8, 0.8, 0.05) == pytest.approx(0.019, abs=0.001)
    assert block(5, 5, 0.8, 0.8, 0.001) < 0.0005
    # A total failure is always caught at five trials a side.
    assert block(5, 5, 1.0, 0.0, 0.001) == pytest.approx(1.0)


def test_run_level_false_alarm_stays_under_five_percent_with_ten_flaky_tasks() -> None:
    q = gate_error_rates.block_probability(5, 5, 0.8, 0.8, 0.05 / 50)
    assert 1 - (1 - q) ** 10 < 0.05
    uncorrected = gate_error_rates.block_probability(5, 5, 0.8, 0.8, 0.05)
    assert 1 - (1 - uncorrected) ** 10 > 0.15  # what --multiplicity none would give


def test_trials_to_decide_a_total_failure() -> None:
    needed = gate_error_rates.trials_to_detect_total_failure
    # A baseline of one stored trial is nearly powerless now that its size
    # is no longer inflated to match the check: seven trials to decide a
    # total failure on its own, fifteen once two tests share alpha.
    assert needed(1, 0.05) == 7 and needed(1, 0.025) == 15
    assert needed(5, 0.025) == 2 and needed(20, 0.001) == 3


def test_main_prints_the_tables(capsys: pytest.CaptureFixture[str]) -> None:
    assert gate_error_rates.main([]) == 0
    out = capsys.readouterr().out
    assert "### Per-task false alarm on an unchanged flaky task" in out
    assert "| 1.0 -> 0.4 | 5 | 5 | 68.3 % | 33.7 % | 7.8 % | 7.8 % |" in out
    assert "### Check trials needed to decide a total failure" in out
