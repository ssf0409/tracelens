#!/usr/bin/env python3
"""Error rates of the ``tracelens compare`` verdict, by seeded simulation.

Prints the Markdown table that ``docs/statistical-contract.md``
("Run-versus-run comparison") quotes. Every cell runs the real
``paired_task_effect`` and ``decide`` over simulated per-task paired
differences at the command's defaults (threshold 0.03, confidence 0.95,
B = 10000) and reports how often the verdict is a regression:

- *no change*: each task's difference is drawn around zero, so every
  regression verdict is a false alarm;
- *a real drop*: every task is worse, so a regression verdict is a hit.

Two models of a task's difference are used: a normal difference with
standard deviation 0.1 (the null simulation in issue #112), and a pass/fail
task observed five times a side with its pass probability drawn from
U(0.1, 0.9). The "interval alone" column is the verdict table before #112,
which called a regression whenever the interval excluded zero.

    uv run --no-sync python scripts/compare_error_rates.py
    uv run --no-sync python scripts/compare_error_rates.py --runs 500   # quicker look

Regenerate the table whenever the verdict rule or a default changes, and
paste the output into the contract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import numpy as np

from tracelens.statistics.run_comparison import (
    DEFAULT_THRESHOLD,
    PairedEffect,
    Verdict,
    decide,
    excludes_zero,
    min_tasks_for,
    paired_task_effect,
)

TASKS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)
RUNS = 2000
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
SD = 0.1
TRIALS = 5

Draw = Callable[[np.random.Generator, int], np.ndarray]


def normal(shift: float) -> Draw:
    """Per-task paired differences drawn from N(shift, SD^2)."""

    def draw(rng: np.random.Generator, tasks: int) -> np.ndarray:
        return rng.normal(shift, SD, size=tasks)

    return draw


def pass_fail(shift: float) -> Draw:
    """Pass/fail tasks, ``TRIALS`` a side, the candidate's pass probability shifted."""

    def draw(rng: np.random.Generator, tasks: int) -> np.ndarray:
        p = rng.uniform(0.1, 0.9, size=tasks)
        baseline = rng.binomial(TRIALS, p) / TRIALS
        candidate = rng.binomial(TRIALS, np.clip(p + shift, 0.0, 1.0)) / TRIALS
        return np.asarray(candidate - baseline, dtype=float)

    return draw


SCENARIOS: dict[str, Draw] = {
    "no change": normal(0.0),
    "drop of 0.10": normal(-0.10),
    "pass/fail, no change": pass_fail(0.0),
    "pass/fail, 20-point drop": pass_fail(-0.20),
}


def interval_alone(effect: PairedEffect, threshold: float) -> bool:
    """The regression row before #112: the interval excludes 0 and delta <= -threshold."""
    if effect.delta is None or effect.ci_lower is None or effect.ci_upper is None:
        return False
    return excludes_zero(effect.ci_lower, effect.ci_upper) and effect.delta <= -threshold


def regression_rates(
    scenario: str,
    tasks: int,
    *,
    runs: int = RUNS,
    n_bootstrap: int = N_BOOTSTRAP,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, float]:
    """How often (the verdict, the interval alone) call a regression.

    Seeded by the scenario and the task count, so every cell is reproducible
    on its own.
    """
    draw = SCENARIOS[scenario]
    rng = np.random.default_rng([112, tasks, list(SCENARIOS).index(scenario)])
    verdict_calls = interval_calls = 0
    for i in range(runs):
        effect = paired_task_effect(
            draw(rng, tasks).tolist(), confidence=CONFIDENCE, n_bootstrap=n_bootstrap, seed=i
        )
        verdict_calls += decide(effect, threshold) is Verdict.REGRESSION
        interval_calls += interval_alone(effect, threshold)
    return verdict_calls / runs, interval_calls / runs


HEADER = (
    "tasks T",
    "no change, interval alone",
    "no change",
    "drop of 0.10",
    "pass/fail, no change",
    "pass/fail, 20-point drop",
)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def print_table(runs: int, n_bootstrap: int) -> None:
    floor = min_tasks_for(CONFIDENCE)
    standard_error = (0.025 * 0.975 / runs) ** 0.5
    print(
        "**Error rates.** How often the verdict is a regression, out of "
        f"{runs} simulated comparisons per cell (threshold {DEFAULT_THRESHOLD}, "
        f"confidence {CONFIDENCE}, B = {n_bootstrap}; `scripts/compare_error_rates.py` "
        "regenerates this table). *No change* draws each task's paired difference "
        f"from a normal distribution around 0 with standard deviation {SD}, and "
        "*drop of 0.10* around -0.10; the pass/fail columns observe each task "
        f"{TRIALS} times a side at a pass probability drawn from U(0.1, 0.9), which "
        "the drop lowers by 0.20. *Interval alone* is the verdict before #112. "
        f"Below {floor} tasks there is no verdict. Near 2.5 % a cell's sampling "
        f"error is about {200 * standard_error:.1f} percentage points either way "
        "(two standard errors).\n"
    )
    rows = []
    for tasks in TASKS:
        null, null_interval = regression_rates(
            "no change", tasks, runs=runs, n_bootstrap=n_bootstrap
        )
        drop, _ = regression_rates("drop of 0.10", tasks, runs=runs, n_bootstrap=n_bootstrap)
        pf_null, _ = regression_rates(
            "pass/fail, no change", tasks, runs=runs, n_bootstrap=n_bootstrap
        )
        pf_drop, _ = regression_rates(
            "pass/fail, 20-point drop", tasks, runs=runs, n_bootstrap=n_bootstrap
        )
        rows.append(
            [str(tasks), pct(null_interval), pct(null), pct(drop), pct(pf_null), pct(pf_drop)]
        )
    print(table(HEADER, rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--runs", type=int, default=RUNS, help=f"comparisons per cell (default {RUNS})")
    parser.add_argument(
        "--bootstrap", type=int, default=N_BOOTSTRAP, dest="n_bootstrap",
        help=f"bootstrap resamples and sign-flip draws (default {N_BOOTSTRAP})",
    )
    args = parser.parse_args(argv)
    print_table(args.runs, args.n_bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
