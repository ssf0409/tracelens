#!/usr/bin/env python3
"""Error rates of the baseline gate's per-task test, by exact enumeration.

Prints the Markdown tables that ``docs/statistical-contract.md`` ("Baseline
regression detection") and ``docs/ci-cd-integration.md`` ("What the gate
can detect") quote. Every number is a sum over the possible outcomes of two
binomial samples, decided by the real ``RegressionDetector`` at the per-test
level the gate uses: ``alpha / T`` under Holm over a family of ``T``
compared (task, metric) tests.

    uv run --no-sync python scripts/gate_error_rates.py            # the tables
    uv run --no-sync python scripts/gate_error_rates.py --simulate # plus a seeded
                                                                   # Monte Carlo run
                                                                   # through evaluate_gate

The Monte Carlo part reproduces the run-level claims end to end (the null
false-alarm rate of a suite with flaky tasks, and the suite-level criterion
on a broad regression). Regenerate the tables whenever the test or the
default policy changes, and paste the output into the two pages.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import cache
from math import comb

from tracelens.baselines.comparison import RegressionDetector, RegressionSeverity
from tracelens.baselines.manager import TaskBaseline

ALPHA = 0.05
# Holm holds the smallest p-value of T tasks to alpha / T.
FAMILY_SIZES = (1, 2, 10, 50)
SIZES = ((5, 5), (10, 5), (20, 5), (10, 10), (20, 20))
FLAKY_RATES = (0.9, 0.8, 0.5)
REGRESSIONS = ((1.0, 0.4), (1.0, 0.6), (1.0, 0.0), (0.8, 0.4))

# Only the p-value is read: skip the per-finding "trials needed" scans.
_detector = RegressionDetector(power_notes=False)


@cache
def p_value(k_b: int, n_b: int, k_c: int, n_c: int) -> float | None:
    """The gate's one-sided p-value for a drop from k_b/n_b to k_c/n_c.

    ``None`` when the detector reports nothing (no drop, or a drop below
    its reporting floor), which never blocks.
    """
    baseline = TaskBaseline(task_id="t")
    baseline.add_metric("pass_rate", k_b / n_b, sample_size=n_b)
    trials = [{"pass_rate": 1.0 if i < k_c else 0.0} for i in range(n_c)]
    report = _detector.compare(baseline, trials)
    if not report.regressions:
        return None
    finding = report.regressions[0]
    if finding.severity is RegressionSeverity.NONE:
        return None
    return finding.p_value


def binomial(k: int, n: int, p: float) -> float:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def block_probability(n_b: int, n_c: int, p_b: float, p_c: float, level: float) -> float:
    """P(the task blocks at ``level``) when its true rates are p_b then p_c."""
    total = 0.0
    for k_b in range(n_b + 1):
        weight_b = binomial(k_b, n_b, p_b)
        if weight_b == 0.0:
            continue
        for k_c in range(n_c + 1):
            weight_c = binomial(k_c, n_c, p_c)
            if weight_c == 0.0:
                continue
            p = p_value(k_b, n_b, k_c, n_c)
            if p is not None and p <= level:
                total += weight_b * weight_c
    return total


def trials_to_detect_total_failure(n_b: int, level: float, cap: int = 60) -> int | None:
    for n_c in range(1, cap + 1):
        p = p_value(n_b, n_b, 0, n_c)
        if p is not None and p <= level:
            return n_c
    return None


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def print_tables() -> None:
    levels = [(t, ALPHA / t) for t in FAMILY_SIZES]
    print("### Per-task false alarm on an unchanged flaky task\n")
    print(
        "Probability that one task whose true pass rate is `p` on both sides blocks "
        "by chance, per test level `alpha / T` (Holm over a family of `T` compared (task, metric) tests).\n"
    )
    rows = []
    for p in FLAKY_RATES:
        for n_b, n_c in SIZES:
            rows.append([f"{p:.1f}", str(n_b), str(n_c)] + [
                pct(block_probability(n_b, n_c, p, p, level)) for _t, level in levels
            ])
    print(table(["p", "baseline n", "check n"] + [f"T={t}" for t, _ in levels], rows))
    print()
    print("### Run-level false alarm with F flaky tasks (p = 0.8, baseline 5, check 5)\n")
    print("`1 - (1 - q)^F` with `q` from the row above; deterministic tasks add nothing. This is the whole run-level rate: the suite criterion does not block by default.\n")
    rows = []
    for t, level in levels:
        q = block_probability(5, 5, 0.8, 0.8, level)
        rows.append([f"T={t}"] + [pct(1 - (1 - q) ** f) for f in (1, 5, 10, 20, 50)])
    print(table(["Holm family", "F=1", "F=5", "F=10", "F=20", "F=50"], rows))
    print()
    print("### Power: one task regresses, the rest are unchanged\n")
    print("Probability that the regressed task blocks, per test level `alpha / T`.\n")
    rows = []
    for p_b, p_c in REGRESSIONS:
        for n_b, n_c in SIZES:
            rows.append([f"{p_b:.1f} -> {p_c:.1f}", str(n_b), str(n_c)] + [
                pct(block_probability(n_b, n_c, p_b, p_c, level)) for _t, level in levels
            ])
    print(table(["drop", "baseline n", "check n"] + [f"T={t}" for t, _ in levels], rows))
    print()
    print("### Check trials needed to decide a total failure\n")
    print("Smallest check size at which 0 passes after a perfect baseline reaches the level.\n")
    rows = []
    for n_b in (1, 2, 3, 5, 10, 20):
        rows.append([str(n_b)] + [
            str(trials_to_detect_total_failure(n_b, level) or ">60") for _t, level in levels
        ])
    print(table(["baseline n"] + [f"T={t}" for t, _ in levels], rows))
    print()


def simulate(runs: int, seed: int) -> None:
    """Seeded Monte Carlo through ``evaluate_gate`` for the run-level claims."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from tracelens.baselines.manager import BaselineManager
    from tracelens.core.outcome import Outcome
    from tracelens.core.trial import Trial, TrialBatch, TrialStatus
    from tracelens.reporting.gate import GateStatus, evaluate_gate

    rng = np.random.default_rng(seed)

    def run_once(root: Path, tasks: int, flaky: int, n: int, p_flaky: float,
                 drop: tuple[int, float] | None) -> GateStatus:
        manager = BaselineManager(root / "baselines.json")
        batch = TrialBatch()
        for index in range(tasks):
            task_id = f"t{index:03d}"
            if index < flaky:
                passes_b, rate_c = int(rng.binomial(n, p_flaky)), p_flaky
            else:
                passes_b, rate_c = n, 1.0
            if drop is not None and index < drop[0]:
                passes_b, rate_c = n, drop[1]
            baseline = TaskBaseline(task_id=task_id)
            baseline.add_metric("pass_rate", passes_b / n, sample_size=n)
            manager.set_baseline(baseline)
            passes_c = int(rng.binomial(n, rate_c))
            for run_index in range(n):
                trial = Trial(task_id=task_id, run_index=run_index, status=TrialStatus.COMPLETED)
                passed = run_index < passes_c
                trial.add_outcome(Outcome(
                    trial_id=trial.trial_id, grader_id="g", passed=passed,
                    score=1.0 if passed else 0.0,
                ))
                batch.add_trial(trial)
        return evaluate_gate(batch, manager).status

    scenarios = [
        ("null, T=50, 20% flaky at 0.8, n=5", dict(tasks=50, flaky=10, n=5, p_flaky=0.8, drop=None)),
        ("null, T=100, all flaky at 0.8, n=5", dict(tasks=100, flaky=100, n=5, p_flaky=0.8, drop=None)),
        ("one task 1.0 -> 0.4, T=1, n=5", dict(tasks=1, flaky=0, n=5, p_flaky=0.8, drop=(1, 0.4))),
        ("one task 1.0 -> 0.4, T=1, n=10", dict(tasks=1, flaky=0, n=10, p_flaky=0.8, drop=(1, 0.4))),
        ("one task 1.0 -> 0.4, T=50, n=10", dict(tasks=50, flaky=0, n=10, p_flaky=0.8, drop=(1, 0.4))),
        ("broad: 20 of 50 tasks 1.0 -> 0.8, n=5", dict(tasks=50, flaky=0, n=5, p_flaky=0.8, drop=(20, 0.8))),
    ]
    print(f"### Monte Carlo through evaluate_gate ({runs} runs each, seed {seed})\n")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, kwargs in scenarios:
            blocked = 0
            for i in range(runs):
                root = Path(tmp) / label.replace(" ", "_") / str(i)
                root.mkdir(parents=True)
                blocked += run_once(root, **kwargs) is GateStatus.BLOCKED  # type: ignore[arg-type]
            rows.append([label, pct(blocked / runs)])
    print(table(["scenario", "runs blocked"], rows))
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--simulate", action="store_true", help="also run the Monte Carlo scenarios")
    parser.add_argument("--runs", type=int, default=200, help="Monte Carlo runs per scenario")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print_tables()
    if args.simulate:
        simulate(args.runs, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
