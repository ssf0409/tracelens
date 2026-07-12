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

  pass_rate: 0.9200 -> 0.8400 (-8.7%)
should_block_ci= True
```

That is the CI gate: a pull request can exit non-zero when
`report.should_block_ci(threshold=RegressionSeverity.MODERATE)` returns `True`.

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
misconfigured gate (no `--baselines-file`, or a missing/unparseable file —
checked before the eval runs). The gate always prints a summary line
(`N checked, M skipped (no baseline), B blocking regression(s)`) plus per-task
warnings for missing baselines.

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
