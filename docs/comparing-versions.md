# Comparing Versions

You changed the model, or you rewrote the prompt. The eval suite that was passing
at 38% is now at 78%. Two questions follow immediately:

1. **Did behavior actually change?** Or is this run-to-run jitter from a
   non-deterministic agent?
2. **Is the change real or noise?** A 40-point jump is obviously real. A
   3-point jump on 20 trials usually isn't. Where's the line?

This page answers both with one workflow: run the *same* eval set against two
configurations, stamp each with a `DecisionSpec` so results are attributable,
and use a bootstrap comparison to decide whether the difference is statistically
real.

The full runnable example is
[`examples/version_compare.py`](https://github.com/ssf0409/tracelens/blob/main/examples/version_compare.py).
It uses a seeded, simulated agent so it runs with no API keys; swap in your real
adapter and the comparison machinery is identical.

## The setup: one ruler, two things to measure

The cardinal rule of a fair comparison is to **change only the thing under
test**. There are two valid axes:

- **Model change** — same prompt, swap `ModelConfig` (e.g. `gpt-4o-mini` →
  `gpt-4o`).
- **Prompt change** — same model, swap the prompt text in `PromptSpec`.

Everything else stays fixed, and critically, the **grader and the eval set are
the ruler** — they must be identical across both versions. If you change the
grader and the prompt at the same time, you can't attribute the delta to either.
What you vary is the **adapter** (the agent invocation) and the **`DecisionSpec`**
(the attributable description of what produced the result).

## Step 1: parameterize and stamp each version

Wire one adapter per configuration, and build a `DecisionSpec` that records the
model and prompt behind it. `ModelConfig` requires a `provider`; `model_id` is
the version you're pinning. `PromptSpec.from_prompts` hashes the prompt text so
two different prompts produce two different fingerprints without storing the raw
text in your baselines.

```python
from tracelens import DecisionSpec, ModelConfig, PromptSpec

def spec_for(version: str, prompt_text: str) -> DecisionSpec:
    return DecisionSpec(
        model=ModelConfig(provider="openai", model_id="gpt-4o-mini", temperature=0.7),
        prompts=PromptSpec.from_prompts(system_prompt=prompt_text, prompt_version=version),
    )

v1_spec = spec_for("v1", "Reply to the support ticket.")
v2_spec = spec_for(
    "v2", "Reply concisely, cite the relevant policy, and propose concrete next steps."
)
```

The two specs now have different fingerprints because the prompt hashes differ.
That difference is what makes a result *attributable*: a transcript stamped with
`v2_spec.fingerprint` provably came from the v2 prompt and not from v1.

```python
print(v1_spec.fingerprint_short)            # 20a1b674339b
print(v2_spec.fingerprint_short)            # cc545403c02b
print(v1_spec.fingerprint != v2_spec.fingerprint)  # True
```

If you change the model instead, swap the `ModelConfig` and the fingerprint moves
the same way. Either way, each result carries a fingerprint of exactly the
configuration that produced it. See [Reproducibility & DecisionSpec](reproducibility.md)
for what goes into a fingerprint and what's deliberately left out.

## Step 2: run both versions on the same eval set

Pass the spec to `EvaluationRunner` via `decision_spec=`; the runner stamps it
onto every transcript that doesn't already carry one. Use `num_runs > 1` — a
single run per task tells you nothing about a non-deterministic agent's variance.

```python
from tracelens import EvaluationRunner, RunnerConfig, SimpleAdapter

async def run_version(make_adapter, spec: DecisionSpec):
    runner = EvaluationRunner(
        make_adapter(),
        [ReplyQualityGrader("reply_quality")],   # SAME grader for both versions
        RunnerConfig(num_runs=10, max_concurrency=1),
        decision_spec=spec,
    )
    batch = await runner.run(TASKS)              # SAME eval set for both versions
    scores = [o.score for trial in batch.trials for o in trial.outcomes]
    return batch, scores

b1, s1 = await run_version(lambda: SimpleAdapter(make_agent(0.66)), v1_spec)
b2, s2 = await run_version(lambda: SimpleAdapter(make_agent(0.82)), v2_spec)
```

`scores` is the flattened list of per-trial grader scores (6 tasks × 10 runs =
60 values per version). That per-trial granularity is what the significance test
in the next step consumes.

Printing the per-version summary gives you the headline numbers:

```
version comparison
------------------
  v1 [20a1b674339b]  pass_rate=38%  mean_quality=0.656  n=60
  v2 [cc545403c02b]  pass_rate=78%  mean_quality=0.816  n=60
```

The fingerprint in brackets is your audit trail: every row is tied to a specific
model + prompt.

## Step 3: the verdict — `tracelens compare`

The means say v2 looks better. The comparison tells you whether to believe it,
and whether it matters. With both runs saved (`tracelens run --save-trials`),
it is one command:

```bash
tracelens compare v1-trials.json v2-trials.json --metric mean_score --threshold 0.05
```

or, in Python, `compare_runs(b1, b2, metric="mean_score", threshold=0.05)` on
the two `TrialBatch` objects (this is what the example does). Either way the
statistics are the ones fixed in the
[statistical contract](statistical-contract.md#run-versus-run-comparison-tracelens-compare-issue-28):

- **The task is the sampling unit.** Each task's mean score under v1 and under
  v2 is paired, and the interval comes from resampling *tasks*. Six tickets of
  different difficulty therefore cancel out instead of looking like noise, and
  ten repeated trials of one ticket never masquerade as ten independent samples.
- **Comparability is checked, not assumed.** The runs' provenance must show the
  same task content and graders; a task edited between the runs, or a changed
  grader, is refused rather than matched by id.
- **Three readings, one verdict.** Significance (does the interval exclude 0),
  practical relevance (`|delta|` against `--threshold`), and evidence (is the
  interval narrow enough to say anything). The verdict is one of *improvement*,
  *regression*, *significant but below the threshold*, *equivalent within the
  threshold*, *inconclusive*, or *insufficient evidence*; exit codes are 0, 1,
  and 2 respectively for "no regression", "regression", and "cannot tell".

The example prints the same summary the command does:

```text
Compared v2 vs v1 on mean_score (higher is better): paired task bootstrap over 6 task(s)
  v1: 60 trials, 60 gradable, 6 task(s) with values
  v2: 60 trials, 60 gradable, 6 task(s) with values
  tasks: 6 task(s) compared, aligned by content
  delta = +0.1599  95% CI [+0.1213, +0.1964]  p = 0.0312 (exact)  (B = 10000, seed = 0)
  readings: significant, |delta| >= threshold 0.05
  Verdict: IMPROVEMENT (exit 0)
  What changed: DecisionSpec prompts (attribution evidence, not proof of cause)
  What moved (largest first): ticket-4 +0.230 (n 10/10), ticket-1 +0.188 (n 10/10), ...
```

Read it top to bottom: *what was compared* (metric, tasks, how they were
aligned), *what the data say* (delta, interval, p-value, and the three
readings), *the verdict*, then *what changed* (from the two `DecisionSpec`s;
here only the prompts) next to *what moved* (the tasks with the largest paired
differences, the ones to read first). "What changed" is attribution evidence,
not proof of cause.

Every field in that summary is also in `--output compare.json`, so a CI job
can branch on the exit code and archive the record. An inconclusive comparison
exits 2 on purpose: not enough evidence must never read as "no regression";
pass `--observe` for dashboards that only want the numbers.

### Trial-level comparison with `compare_metrics`

`compare_metrics` is the older, lower-level tool: it compares two flat lists of
values with an *unpaired* bootstrap and treats every value as an independent
sample. Use it when the two samples really are independent draws (two unrelated
sets of trials, latency samples), not for two runs of the same eval set, where
`compare_runs` pairs the tasks. It returns a `ComparisonResult`:

```python
from tracelens.statistics.inference import compare_metrics

res = compare_metrics(s1, s2, confidence=0.95, compute_p_value=True)
```

`compute_p_value=True` adds a permutation test (it's off by default because it's
the expensive part). The result fields:

| Field | Meaning |
| --- | --- |
| `delta` | `current.mean - baseline.mean` (here, v2 − v1) |
| `relative_delta` | `delta / |baseline.mean|` |
| `ci_lower`, `ci_upper` | bootstrap 95% CI **of the difference** |
| `is_significant` | `True` when the CI excludes 0 |
| `cohens_d` | effect size (standardized magnitude) |
| `p_value` | permutation-test p-value (`None` unless `compute_p_value=True`) |

On the example's two score lists this reads:

```
  quality delta (v2 - v1) = +0.160  95% CI [+0.119, +0.201]  cohens_d=1.38  p=0.000
```

How to read this, in order of what matters:

- **`95% CI [+0.119, +0.201]` excludes 0** → the difference is real, not noise.
  This is the single most important line. If the CI had been `[-0.02, +0.34]`,
  the improvement would be *plausibly zero* and you should not ship on it. The
  `is_significant` flag is exactly this check.
- **`cohens_d=1.38`** → the effect is *large* (Cohen's conventions:
  <0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large; exposed as
  `res.effect_magnitude`). A change can be statistically significant but tiny;
  `d` tells you whether it's worth caring about. 1.38 means the two distributions
  barely overlap.
- **`p=0.000`** → the permutation test agrees the distributions differ. Treat the
  CI as primary and the p-value as corroboration, not the other way around.

Convenience properties wrap the same logic: `res.is_improvement` (significant
and positive) and `res.is_regression` (significant and negative) read better than
hand-checking `is_significant and delta > 0`.

If the CI had straddled zero, the honest verdict is "no detectable difference at
this sample size" — collect more runs before concluding either way. The full
statistical toolkit is in [Statistical Comparison](statistical-comparison.md).

## Step 4: don't stop at the mean — capability vs reliability

A version that wins on `mean_quality` can still be *flakier*. The mean hides the
distribution. Look at both axes per version:

- **pass@k (capability)** — can this version solve the task *at all* within k
  attempts? A higher pass@k means a higher ceiling.
- **pass^k (reliability)** — does it solve the task on *every one* of k attempts?
  A version with a better mean but a worse pass^k is more capable on a good day
  and less trustworthy in production.

Compute these per version and compare them side by side, not just the means. See
[pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) for the full treatment.

## Two ways to run a comparison

This page covers **ad-hoc head-to-head**: you have two configs in hand right now
and you want a verdict. That's `tracelens compare` on two saved runs (or
`compare_runs` on two batches).

The other mode is **baseline-gated comparison for CI**: you store a baseline once,
then every future run is compared against it automatically and the build fails on
a regression. That's the job of `RegressionDetector` — see the
[Baseline Regression Tutorial](baseline-regression-tutorial.md).

The two modes share the `DecisionSpec` machinery. `RegressionDetector.compare_with_specs`
takes both runs' specs and can distinguish *"the model or prompt changed"* (an
intentional, attributable difference) from *"only infra changed"* (resource
budget, concurrency — noise you don't want to gate on). That spec-aware comparison
is demonstrated in
[`examples/noise_aware_regression.py`](https://github.com/ssf0409/tracelens/blob/main/examples/noise_aware_regression.py),
and explained in [Reproducibility & DecisionSpec](reproducibility.md).

## See also

- [Statistical Contract](statistical-contract.md#run-versus-run-comparison-tracelens-compare-issue-28)
  — the estimand, sampling unit, and verdict table `tracelens compare` implements.
- [Statistical Comparison](statistical-comparison.md) — the bootstrap CI, effect
  size, and significance machinery behind `compare_metrics`.
- [Reproducibility & DecisionSpec](reproducibility.md) — how fingerprints make a
  result attributable to a specific model/prompt.
- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — capability vs reliability, the
  second dimension every comparison needs.
- [Baseline Regression Tutorial](baseline-regression-tutorial.md) — the CI-gated
  comparison mode and `RegressionDetector`.
