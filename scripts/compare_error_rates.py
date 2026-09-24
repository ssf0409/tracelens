#!/usr/bin/env python3
"""Error rates of the ``tracelens compare`` verdict, by seeded simulation.

Prints the Markdown tables that ``docs/statistical-contract.md``
("Run-versus-run comparison") quotes. Every cell runs the real
``paired_task_effect`` and ``decide`` over simulated per-task paired
differences and reports how often the verdict is a regression (first
table) or exits 0 (second table):

- *no change*: each task's difference is drawn around zero, so every
  regression verdict is a false alarm;
- *a real drop*: every task is worse, so a regression verdict is a hit;
- *a drop of exactly the threshold*: every verdict that exits 0 misses it.

The first table uses two models of a task's difference: a normal difference
with standard deviation 0.1 (the null simulation in issue #112), and a
pass/fail task observed five times a side with its pass probability drawn
from U(0.1, 0.9). The "interval alone" column is the verdict table before
#112, which called a regression whenever the interval excluded zero. The
second table gives each task's difference a standard deviation equal to the
threshold, so a drop of exactly the threshold is hard to tell from a
smaller one.

The comparison runs at the command's own defaults (threshold, confidence,
B), imported from ``tracelens.statistics.run_comparison``, so the tables
describe ``tracelens compare`` as shipped. The grid and the simulation are
fixed below so the tables are reproducible. Every value can be overridden
to look at other settings:

    uv run --no-sync python scripts/compare_error_rates.py
    uv run --no-sync python scripts/compare_error_rates.py --runs 500   # quicker look
    uv run --no-sync python scripts/compare_error_rates.py --tasks 6,12 --confidence 0.99

Regenerate the tables whenever the verdict rule or a default changes, and
paste the output into the contract.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tracelens.statistics.run_comparison import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_THRESHOLD,
    VERDICT_EXIT_CODES,
    PairedEffect,
    Verdict,
    alpha_for,
    decide,
    excludes_zero,
    min_tasks_for,
    paired_task_effect,
)

TASKS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)  # below the 6-task floor, at it, and beyond
RUNS = 2000  # comparisons per cell: about 0.7 points of sampling error near 2.5 %
SD = 0.1  # spread of a task's paired difference: the null simulation in issue #112
TRIALS = 5  # trials a side in the pass/fail model: the num_runs `tracelens init` writes


@dataclass(frozen=True)
class Settings:
    """The comparison's settings and the simulation's."""

    threshold: float = DEFAULT_THRESHOLD
    confidence: float = DEFAULT_CONFIDENCE
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    runs: int = RUNS
    sd: float = SD
    trials: int = TRIALS


DEFAULTS = Settings()
Draw = Callable[[np.random.Generator, int, Settings], np.ndarray]


def normal(shift: float) -> Draw:
    """Per-task paired differences drawn from N(shift, sd^2)."""

    def draw(rng: np.random.Generator, tasks: int, settings: Settings) -> np.ndarray:
        return rng.normal(shift, settings.sd, size=tasks)

    return draw


def pass_fail(shift: float) -> Draw:
    """Pass/fail tasks, ``trials`` a side, the candidate's pass probability shifted."""

    def draw(rng: np.random.Generator, tasks: int, settings: Settings) -> np.ndarray:
        n = settings.trials
        p = rng.uniform(0.1, 0.9, size=tasks)
        baseline = rng.binomial(n, p) / n
        candidate = rng.binomial(n, np.clip(p + shift, 0.0, 1.0)) / n
        return np.asarray(candidate - baseline, dtype=float)

    return draw


def at_threshold_scale(thresholds: float) -> Draw:
    """Differences around ``thresholds`` times the threshold, with the threshold as sd."""

    def draw(rng: np.random.Generator, tasks: int, settings: Settings) -> np.ndarray:
        tau = settings.threshold
        return rng.normal(thresholds * tau, tau, size=tasks)

    return draw


# Each scenario's position seeds its draws: append, never reorder.
SCENARIOS: dict[str, Draw] = {
    "no change": normal(0.0),
    "drop of 0.10": normal(-0.10),
    "pass/fail, no change": pass_fail(0.0),
    "pass/fail, 20-point drop": pass_fail(-0.20),
    "drop of the threshold, sd the threshold": at_threshold_scale(-1.0),
    "no change, sd the threshold": at_threshold_scale(0.0),
}


def interval_alone(effect: PairedEffect, threshold: float) -> bool:
    """The regression row before #112: the interval excludes 0 and delta <= -threshold."""
    if effect.delta is None or effect.ci_lower is None or effect.ci_upper is None:
        return False
    return excludes_zero(effect.ci_lower, effect.ci_upper) and effect.delta <= -threshold


@dataclass(frozen=True)
class Rates:
    """Fractions of the simulated comparisons."""

    regression: float  # the verdict is a regression
    exit_zero: float  # the verdict exits 0
    interval_alone: float  # the verdict before #112 would have been a regression


def simulate(scenario: str, tasks: int, settings: Settings = DEFAULTS) -> Rates:
    """How often the verdict is a regression or exits 0 in ``settings.runs`` comparisons.

    Seeded by the scenario and the task count, so every cell is reproducible
    on its own.
    """
    draw = SCENARIOS[scenario]
    rng = np.random.default_rng([112, tasks, list(SCENARIOS).index(scenario)])
    regression = exit_zero = interval = 0
    for i in range(settings.runs):
        effect = paired_task_effect(
            draw(rng, tasks, settings).tolist(),
            confidence=settings.confidence,
            n_bootstrap=settings.n_bootstrap,
            seed=i,
        )
        verdict = decide(effect, settings.threshold)
        regression += verdict is Verdict.REGRESSION
        exit_zero += VERDICT_EXIT_CODES[verdict] == 0
        interval += interval_alone(effect, settings.threshold)
    runs = settings.runs
    return Rates(regression / runs, exit_zero / runs, interval / runs)


REGRESSION_HEADER = (
    "tasks T",
    "no change, interval alone",
    "no change",
    "drop of 0.10",
    "pass/fail, no change",
    "pass/fail, 20-point drop",
)
EXIT_ZERO_HEADER = ("tasks T", "drop of τ", "no change")


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def sampling_error(rate: float, runs: int) -> float:
    """Two standard errors of a rate near ``rate``, in percentage points."""
    return 200 * math.sqrt(rate * (1 - rate) / runs)


def regression_table(tasks_grid: Sequence[int], settings: Settings = DEFAULTS) -> str:
    s = settings
    floor = min_tasks_for(s.confidence)
    caption = (
        "**Error rates.** How often the verdict is a regression, out of "
        f"{s.runs} simulated comparisons per cell (threshold {s.threshold}, "
        f"confidence {s.confidence}, B = {s.n_bootstrap}; `scripts/compare_error_rates.py` "
        "regenerates this table). *No change* draws each task's paired difference "
        f"from a normal distribution around 0 with standard deviation {s.sd}, and "
        "*drop of 0.10* around -0.10; the pass/fail columns observe each task "
        f"{s.trials} times a side at a pass probability drawn from U(0.1, 0.9), which "
        "the drop lowers by 0.20. *Interval alone* is the verdict before #112. "
        f"Below {floor} tasks there is no verdict. Near 2.5 % a cell's sampling "
        f"error is about {sampling_error(0.025, s.runs):.1f} percentage points either way "
        "(two standard errors)."
    )
    rows = []
    for tasks in tasks_grid:
        null = simulate("no change", tasks, s)
        drop = simulate("drop of 0.10", tasks, s)
        pf_null = simulate("pass/fail, no change", tasks, s)
        pf_drop = simulate("pass/fail, 20-point drop", tasks, s)
        rows.append([
            str(tasks),
            pct(null.interval_alone),
            pct(null.regression),
            pct(drop.regression),
            pct(pf_null.regression),
            pct(pf_drop.regression),
        ])
    return caption + "\n\n" + table(REGRESSION_HEADER, rows)


def exit_zero_table(tasks_grid: Sequence[int], settings: Settings = DEFAULTS) -> str:
    s = settings
    caption = (
        "**Exit 0.** How often the verdict exits 0, out of the same number of "
        "comparisons at the same settings, when each task's paired difference is "
        "drawn from a normal distribution whose standard deviation is the threshold, "
        f"{s.threshold:g}: around -{s.threshold:g} (*drop of τ*, a regression of exactly the "
        "threshold, which every exit 0 misses) or around 0 (*no change*, where exit 0 "
        "is right). Near 5 % a cell's sampling error is about "
        f"{sampling_error(0.05, s.runs):.1f} percentage points either way."
    )
    rows = []
    for tasks in tasks_grid:
        drop = simulate("drop of the threshold, sd the threshold", tasks, s)
        unchanged = simulate("no change, sd the threshold", tasks, s)
        rows.append([str(tasks), pct(drop.exit_zero), pct(unchanged.exit_zero)])
    return caption + "\n\n" + table(EXIT_ZERO_HEADER, rows)


def print_tables(tasks_grid: Sequence[int], settings: Settings = DEFAULTS) -> None:
    print(regression_table(tasks_grid, settings))
    print()
    print(exit_zero_table(tasks_grid, settings))


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text}")
    return value


def task_list(text: str) -> tuple[int, ...]:
    return tuple(positive_int(part) for part in text.split(","))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--tasks", type=task_list, default=TASKS, metavar="T,T,...",
        help="the table's rows: comma-separated task counts (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=positive_int, default=RUNS,
        help="comparisons per cell (default: %(default)s)",
    )
    parser.add_argument(
        "--bootstrap", type=positive_int, default=DEFAULT_N_BOOTSTRAP, dest="n_bootstrap",
        metavar="B", help="bootstrap resamples and sign-flip draws (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold", type=positive_float, default=DEFAULT_THRESHOLD,
        help="the practical threshold τ (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="confidence level of the comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--sd", type=positive_float, default=SD,
        help="standard deviation of a task's difference in the normal model (default: %(default)s)",
    )
    parser.add_argument(
        "--trials", type=positive_int, default=TRIALS,
        help="trials a side in the pass/fail model (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        alpha_for(args.confidence)
    except ValueError as exc:
        parser.error(str(exc))
    settings = Settings(
        threshold=args.threshold,
        confidence=args.confidence,
        n_bootstrap=args.n_bootstrap,
        runs=args.runs,
        sd=args.sd,
        trials=args.trials,
    )
    print_tables(args.tasks, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
