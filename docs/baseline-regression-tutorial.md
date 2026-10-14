# Baseline Regression Tutorial

This tutorial takes you from a first passing TraceLens eval to a baseline
comparison that can block CI. It uses a tiny pass-rate example so the baseline
lifecycle is clear before you wire it into a real agent.

## What You Will Build

By the end, you will have:

- a first passing eval run and report;
- a stored baseline in JSON;
- a candidate run compared against that baseline;
- an intentional failing candidate with expected regression output;
- a promotion path for approved improvements;
- CI wiring that blocks on moderate or severe regressions.

## 1. Run A First Passing Eval

Start with the repository hello-world eval:

```bash
python examples/hello_world.py
tracelens report --results examples/reports/hello_world_report.json --format markdown
```

Expected high-level result:

```text
trials run : 9
pass rate  : 100%
report json: examples/reports/hello_world_report.json
sample md  : examples/reports/hello_world_report.md
```

The checked-in sample report is
[`examples/reports/hello_world_report.md`](https://github.com/ssf0409/tracelens/blob/main/examples/reports/hello_world_report.md).
It shows the pieces CI usually needs: pass rate, pass@k, pass^k, baseline
comparison, regression result, and CI summary.

## 2. Choose Canary Or Capability Baselines

Use the baseline type to encode how cautious promotion should be.

| Baseline type | Use it for | Example | Promotion rule |
|---------------|------------|---------|----------------|
| `CANARY` | A protected floor that should not drift automatically. | A safety task must keep `safety_score >= 0.95`. | Manual promotion only after review. |
| `CAPABILITY` | A moving benchmark for what the agent can currently do. | A math agent currently has `pass_rate = 0.92`. | Auto-promote only when the new run improves enough and has enough samples. |

Concrete rule of thumb: if a drop means "do not ship," use a canary. If an
improvement means "record the new normal," use a capability baseline.

## 3. Create And Store A Baseline

Create a capability baseline for the hello-world style pass-rate metric and a
protected canary for a safety score:

```python
from tracelens.baselines import BaselineManager, PromotionPolicy

manager = BaselineManager("eval/baselines/baselines.json")

manager.create_capability_baseline(
    task_id="math_agent",
    task_name="Math agent pass-rate benchmark",
    metrics={"pass_rate": 0.92},
    metric_stds={"pass_rate": 0.02},
    sample_size=20,
    promotion_policy=PromotionPolicy(
        allow_auto_promotion=True,
        min_improvement_relative=0.02,
        min_samples=20,
    ),
)

manager.create_canary_baseline(
    task_id="safety_contract",
    task_name="Critical safety response contract",
    metrics={"safety_score": 0.95},
    metric_stds={"safety_score": 0.01},
    sample_size=20,
    fingerprint="model-v1-prompt-a-tools-2026-05-24",
)

manager.save()
```

All `BaselineManager` write APIs also accept `decision_spec=`. For canaries,
`fingerprint` is optional when a spec is given — it is derived from
`decision_spec.fingerprint` — and the stored spec is what enables noise-aware
comparison (`compare_with_specs()`) in the CI gate. A hand-typed `fingerprint=`
string still works if you don't build specs.

Commit the generated `eval/baselines/baselines.json` file. Treat it like a test
fixture: changes should be reviewed, and the PR should explain why the baseline
is being created or promoted.

## 4. Compare A Passing Candidate

A candidate run is just a list of metric dictionaries. A real integration would
derive these values from `TrialBatch` outcomes, but the comparison API does not
care where the metrics came from.

```python
from tracelens.baselines import BaselineManager, RegressionDetector

manager = BaselineManager("eval/baselines/baselines.json")
baseline = manager.get_baseline("math_agent")

candidate_results = [
    {"pass_rate": 0.93},
    {"pass_rate": 0.94},
    {"pass_rate": 0.92},
    {"pass_rate": 0.93},
]

report = RegressionDetector(min_delta_percent=5.0).compare(
    baseline,
    candidate_results,
)

print(report.to_ci_output())
print(f"should_block_ci={report.should_block_ci()}")
```

Expected output:

```text
No regressions detected
should_block_ci=False
```

## 5. Run An Intentional Failing Candidate

Now make the candidate worse by more than the default moderate threshold.

```python
from tracelens.baselines import BaselineManager, RegressionDetector, RegressionSeverity

manager = BaselineManager("eval/baselines/baselines.json")
baseline = manager.get_baseline("math_agent")

candidate_results = [{"pass_rate": 0.84}] * 20

report = RegressionDetector(min_delta_percent=5.0).compare(
    baseline,
    candidate_results,
)

print(report.to_ci_output())
print(
    "should_block_ci=",
    report.should_block_ci(threshold=RegressionSeverity.MODERATE),
)
```

Expected regression output:

```text
REGRESSION DETECTED [MODERATE]

  pass_rate: 0.9200 -> 0.8400 (-8.7%) [p=0.0000, significant]
should_block_ci= True
```

That is the CI gate: a pull request can exit non-zero when
`report.should_block_ci(threshold=RegressionSeverity.MODERATE)` returns `True`.
Blocking needs both the size of the drop (severity, from the relative
change) and the evidence for it (a one-sided test on the stored summary and
the candidate samples; here Welch's t-test, which uses each side's own
measured spread rather than assuming the two share one). A drop the evidence cannot confirm is still printed, marked `not
significant` with the number of trials that would decide it, and never
blocks; see the [statistical contract](statistical-contract.md#baseline-regression-detection)
for the tests and their error rates.

## 6. Promote An Approved Improvement

Promotion changes the stored baseline. Do it only after deciding the new result
is trustworthy.

```python
from tracelens.baselines import BaselineManager

manager = BaselineManager("eval/baselines/baselines.json")

promoted, reason = manager.try_promote(
    task_id="math_agent",
    current_metrics={"pass_rate": 0.95},
    metric_stds={"pass_rate": 0.01},
    sample_size=25,
    fingerprint="model-v2-prompt-b-tools-2026-05-24",
)

print(promoted, reason)

if promoted:
    manager.save()
```

Expected output:

```text
True Baseline promoted successfully
```

`try_promote` and `force_promote` also accept `decision_spec=`; promotion then
refreshes the stored spec (archiving the previous one in `previous_versions`)
and derives the new fingerprint from it, so the baseline's spec can never drift
from its fingerprint.

For canaries, prefer manual review. `CANARY` baselines deliberately do not
auto-promote; use `force_promote(...)` only after a maintainer approves the new
floor and the associated `DecisionSpec` fingerprint.

## 7. Wire The Comparison Into CI

If you run evals through the CLI, `tracelens run --baseline-check
--baselines-file eval/baselines/baselines.json --fail-on-regression moderate`
does this end to end. Exit codes: 0 = gate passed, 1 = gate blocked (a blocking
regression, or `--require-baselines` with tasks missing baselines), 2 =
misconfigured or unevaluable gate. Missing/unparseable baseline files are
rejected before execution. After execution, zero checked tasks or any
baseline-backed task with no gradable trials or comparable CLI metrics, or
whose content changed since its baseline was stored (`TaskBaseline.task_hash`
no longer matches the run's provenance), makes the gate `UNEVALUABLE`
(exit 2), even if another task regressed.
The gate always prints a summary line
(`N checked, M skipped (no baseline), B blocking regression(s)`) plus per-task
warnings for missing baselines. The same decision is written to the
`--output` JSON as a `gate` object and rendered in the Markdown and HTML
reports, so `tracelens report --results results.json` shows the status,
threshold, per-task regressions, and reasons exactly as the run decided them.
See [Reading the gate](./ci-cd-integration.md#reading-the-gate).

For a hand-rolled Python gate, run your eval harness on pull requests and exit
with a failure when the regression report blocks:

```python
import sys

from tracelens.baselines import BaselineManager, RegressionDetector, RegressionSeverity

manager = BaselineManager("eval/baselines/baselines.json")
baseline = manager.get_baseline("math_agent")
candidate_results = load_candidate_results()  # Your eval harness owns this.

report = RegressionDetector(min_delta_percent=5.0).compare(
    baseline,
    candidate_results,
)

print(report.to_ci_output())

if report.should_block_ci(threshold=RegressionSeverity.MODERATE):
    sys.exit(1)
```

Use the GitHub Actions template in
[`docs/ci-cd-integration.md`](./ci-cd-integration.md#pull-request-workflow) for
checkout, Python setup, dependency installation, artifacts, and PR comments.

Use the sample report at
[`examples/reports/hello_world_report.md`](https://github.com/ssf0409/tracelens/blob/main/examples/reports/hello_world_report.md)
as the shape to aim for when posting CI summaries: include the metric, baseline
value, current value, delta, regression result, and blocking decision.

## Strict metric coverage for frozen replays

`BaselineManager.compare_to_baseline(task_id, metrics, coverage="strict")`
reports the union of metric names. Matched rows retain the usual comparison
fields and add `coverage: "matched"`. A missing row contains `baseline`,
`current`, `regression: None`, and one of:

- `coverage: "missing_from_current"`: the baseline expected a measurement
  that this run did not produce; `current` is `None`.
- `coverage: "missing_from_baseline"`: this run produced an unknown metric;
  `baseline` is `None`.

A missing row has no numerical verdict. A caller must check `coverage` before
reading `regression`; `None` is not a pass. A new metric with value zero is
still unknown, and a missing metric is never filled with zero. Producers of
sparse counters should explicitly emit zero for known reasons that did not
occur, and omit a metric only when they did not measure it.

The default `coverage="intersection"` preserves the previous result dictionary
and skips unmatched metrics. Strict coverage is opt-in, including for CANARY
baselines: a safety floor and a frozen deterministic replay have different
measurement expectations. An absent task baseline still returns
`{"_no_baseline": True}` and must never clear a caller's gate. Empty comparison
sets carry no passing evidence.

To accept a reviewed change to the metric set, update the stored baseline.
`force_promote` and `update_baseline` add a new metric with the run's value, but
neither ever deletes one. A metric that legitimately no longer exists, such as a
failure reason that was fixed, keeps reporting `missing_from_current` until you
remove it: `del baseline.metrics[name]`, then `manager.save()`. Deleting does not
archive the old value in the baseline's history, so record the reason in the
change that edits the baseline file.

`compare_to_baseline_with_ci(..., coverage="strict")` applies the same coverage
rule. An empty sample list counts as missing from current; missing rows carry
`is_regression: None` and the raw current samples or `None`, rather than an
invented estimate. Strict comparisons reject nonfinite measurements.

The runnable [deterministic canary recipe](https://github.com/ssf0409/tracelens/blob/main/examples/deterministic_canary.py)
executes the checked-in `examples/canary_replay.json` task through an adapter,
transcript, and deterministic metric grader. Record the protected baseline once,
then use fresh output directories for checks:

```bash
uv run --frozen python examples/deterministic_canary.py record \
  --baseline /tmp/workflow-canary/baseline.json --output-dir /tmp/workflow-canary/reference
uv run --frozen python examples/deterministic_canary.py check \
  --baseline /tmp/workflow-canary/baseline.json --scenario unchanged \
  --output-dir /tmp/workflow-canary/unchanged
uv run --frozen python examples/deterministic_canary.py check \
  --baseline /tmp/workflow-canary/baseline.json --scenario missing \
  --output-dir /tmp/workflow-canary/missing
```

The unchanged check exits 0. `missing`, `new`, and `empty` exit 1 with explicit
coverage findings. Invalid input exits 2 before replay. Each successful replay
persists `trials.json` and `gate.json`, including metrics, comparisons, task hash,
and run identity. Checking never promotes or overwrites the baseline; recording
refuses an existing baseline. The fixture demonstrates a local workflow, not an
external service integration. Its Python gate policy is separate from
`tracelens run --baseline-check`, which uses `RegressionDetector` and a limited
CLI metric set; these API options do not add CLI gate behavior.
Unbounded derived values (such as relative change from a zero baseline) are
written as JSON `null` and named in the row's `unavailable_fields`; the consumer
decision still uses the actual comparison flags.

## Two-sided change tolerances

One-sided `regression_threshold_absolute` / `regression_threshold_relative`
remain useful for noisy quality metrics. For a frozen deterministic workflow,
either an increase or a decrease can be unexpected. Declare that expectation
on each metric:

```python
from tracelens.baselines import BaselineManager

manager = BaselineManager("eval/baselines/canary-baselines.json")
baseline = manager.create_canary_baseline(
    task_id="fixed-workflow",
    metrics={"accepted": 2.0, "rejected.invalid": 1.0},
    fingerprint="workflow-v1-replay-2026-10-13",
)
for metric in baseline.metrics.values():
    metric.change_tolerance_absolute = 0.0
manager.save()
```

Assign the field, as above, on new and existing metrics alike.
`TaskBaseline.add_metric()` also accepts `change_tolerance_absolute` and
`change_tolerance_relative`, but it replaces the whole metric: every setting it
is not given (`std_deviation`, `sample_size`, regression thresholds,
`higher_is_better`, `is_rate`) returns to its default.

Each configured tolerance is a hard bound:

- Absolute: `abs(current - baseline) > change_tolerance_absolute`.
- Relative: `abs(current - baseline) > change_tolerance_relative * abs(baseline)`.
- Setting both means exceeding either bound reports a change, so the smaller
  allowance wins. This is the opposite of `math.isclose` and `pytest.approx`,
  where the larger one wins: here an absolute bound cannot widen a relative bound
  near zero. Equality is within tolerance. A zero baseline allows no nonzero
  movement under a relative bound; use an absolute bound when movement away from
  zero is acceptable.
- Tolerances must be finite and nonnegative. Zero requires exact numerical
  equality with no hidden float epsilon; use an explicit nonzero tolerance for
  calculated floats that need rounding allowance. A value kept to a fixed
  quantum, such as P&L rounded to 1e-6, can move by one quantum and still exceed
  a tolerance equal to that quantum, because the difference is computed in floats
  (`36.620401 - 36.6204` is `1.00000000458e-06`). To allow one quantum but not
  two, set the tolerance between them, for example `1.5e-6`.

Only metrics with a configured tolerance add `changed: True | False` to their
matched comparison row. The existing `delta`, `relative_change`, `z_score`, and
`regression` are preserved. Old files without these fields load with tolerances
disabled and retain the previous comparison dictionary. `None` disables a bound.
The legacy zero-baseline `relative_change` may still be infinite; the change
check uses the absolute bound formula above and does not depend on that value.

Both comparison methods support tolerances. In `compare_to_baseline_with_ci`,
`changed` compares the current sample **mean** with the stored value. It is
independent of `is_significant` and `is_regression`; opposing sample changes can
cancel in the mean. It does not check each sample, distribution stability, or
statistical equivalence. For a deterministic replay, the point comparison of
observed counts is sufficient. For noisy scores, retain the statistical tests
and their evidence requirements.

`update_baseline(..., keep_thresholds=True)` (the default) and both promotion
paths preserve the tolerance settings and archive them in history.
`keep_thresholds=False` explicitly clears regression thresholds and change
tolerances. CANARY baselines continue to require manual promotion.
Tolerances are carried forward only for metrics the baseline already had. A
metric that promotion or `update_baseline` adds for the first time, such as a
new failure reason accepted with `force_promote`, starts with tolerances
disabled; set its tolerance as part of that review.
Saving rejects invalid assigned tolerances before opening the baseline file,
so serialization cannot turn a nonfinite bound into a disabled `null` setting.

The recipe records zero absolute tolerances on all metrics and uses strict
coverage. Its consumer blocks on either coverage findings, regression, or
`changed`; it does not silently promote a favorable count change:

```bash
uv run --frozen python examples/deterministic_canary.py check \
  --baseline /tmp/workflow-canary/baseline.json --scenario increase \
  --output-dir /tmp/workflow-canary/increase
uv run --frozen python examples/deterministic_canary.py check \
  --baseline /tmp/workflow-canary/baseline.json --scenario decrease \
  --output-dir /tmp/workflow-canary/decrease
```

Both exit 1: accepted requests move from 2 to 3 or 1, with `changed: true` even
though `regression: false` (no one-sided threshold was configured). A metric
without a tolerance has no `changed` verdict and could move freely, so the check
refuses, with exit 2 before replay, a baseline in which any metric lacks one:
a baseline recorded before tolerances existed, or a metric first added by
promotion. Review those metrics and set their tolerances explicitly; the recipe
never rewrites the baseline.
The `zero-channel` scenario activates the existing `rejected.quota` counter from
zero to one; it also exits 1 and persists a gate with unavailable relative change.

## Baseline Review Checklist

Before committing a baseline change, confirm:

- the baseline type matches the risk: `CANARY` for protected floors,
  `CAPABILITY` for moving benchmarks;
- sample count is high enough for the decision;
- the metric direction is correct;
- the candidate was run under the same relevant `DecisionSpec` and
  infrastructure configuration;
- the PR explains whether the change creates, compares, or promotes a baseline;
- CI artifacts link to the report used for the decision.
