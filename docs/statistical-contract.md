# Statistical Contract

Every number TraceLens reports — pass@k, pass^k, pass rate, confidence
intervals, regression verdicts — is a statistic over trials. This page is the
single definition of **what is counted, what the unit of analysis is, and
which estimator is used**, so that the runner, the statistics module, the
reporting layer, and the CLI gate agree.

The rule: an implementation change that alters an estimator, a sampling unit,
or a validity rule updates this page **in the same pull request** and calls
the change out in `CHANGELOG.md`. Where the current code deviates from this
contract, the deviation is listed at the bottom with the issue that resolves
it; a report must not present a deviating number as if it followed the
contract.

## Vocabulary

- **Task** — one evaluation case (`Task`). The population of interest is
  "tasks like the ones in this eval set".
- **Trial** — one execution of one task (`Trial`), identified by
  `(task_id, run_index)`. Repeated trials of the same task are repeated
  measurements of that task, not independent draws from the task population.
- **Outcome** — one grader's verdict on one trial (`Outcome`). A trial can
  carry several outcomes, one per grader.
- **Harness failure** — a trial that says nothing about the agent because the
  evaluation machinery failed: infrastructure errors and grader crashes.
- **Suite statistic** — a number summarising the whole eval set: suite
  pass@k, suite pass^k, overall pass rate, mean score.

## Trial validity

Which trials enter which statistic:

| Trial state | Meaning | Agent statistics | Reported separately as |
|---|---|---|---|
| `COMPLETED`, no grader error | agent finished and was graded | included; passes iff every outcome passed | — |
| `FAILED` | agent-level failure: adapter error not classified as infrastructure, or a teardown failure | included as a **failure** | — |
| `TIMEOUT` | runner time budget exceeded | included as a **failure**; the estimand is "passes within budget" | — |
| `INFRA_ERROR` | infrastructure failure: OOM, network, sandbox, `InfraError` | **excluded** | `infra_error_count` / `infra_error_rate` |
| any status with an outcome where `grader_error=True` | the grading harness crashed | **excluded** | `grader_error_count` / `grader_error_rate` |
| `SKIPPED`, `PENDING`, `RUNNING` | not evidence | excluded | not in any denominator |

An included trial is a **gradable trial**. `passed` is true iff the trial
carries at least one outcome and every outcome passed. Trials that never
produced a transcript (`TIMEOUT`, and `FAILED` during setup or run) have no
outcomes and therefore count as failures. A `FAILED` status caused by a
teardown error after a graded run is judged by its outcomes and flagged with
`metadata["teardown_failed"]`.

Harness failures are never folded into agent failure. They are excluded from
agent statistics and shown next to them with counts, because a spike in
either rate means the evaluation broke, not the agent.

## Sampling units

The sampling unit is the thing whose count is `n` and the thing a bootstrap
resamples. It differs by question:

| Statistic | Unit | Why |
|---|---|---|
| Suite pass@k, suite pass^k, suite mean score, and their confidence intervals | **task** | The claim generalises to "tasks like these". Trials within a task are repeated measurements; they enter through the per-task statistic, not as independent samples. |
| Per-task baseline regression (`RegressionDetector`, `tracelens run --baseline-check`) | **trial**, within one task | Compares one task's current trials against that task's stored baseline distribution. Valid only within the task. |
| Run-versus-run comparison (`tracelens compare`, planned) | **task, paired** | Each task is observed under both runs; the paired per-task difference is the unit. Matching uses task content identity, never `task_id` alone. |

Consequence: a suite-level confidence interval narrows with more *tasks*, not
with more runs per task. More runs per task sharpen each per-task statistic
but do not by themselves justify a narrower suite interval. TraceLens does
not currently model within-task sampling noise in suite intervals (no
hierarchical resampling); read suite intervals as conditional on the
per-task scores.

## Estimators

### pass@k (capability)

Per task, with `n` gradable trials and `c` passes:

```text
pass@k = 1 - C(n - c, k) / C(n, k)
```

This is the unbiased estimator from Chen et al. (2021). It is defined only
for `n >= k`. Suite pass@k is the unweighted mean of per-task pass@k over
**eligible tasks** (`n >= k`), reported with eligible/total task counts.

Contract for `n < k`: the per-task value is **unavailable**, not a fallback
(see *Availability*).

### pass^k (reliability)

Per task, order trials by `run_index` and count windows of `k` consecutive
trials:

```text
pass^k = (windows in which all k trials passed) / (n - k + 1)
```

This is a **consecutive-window statistic**. It is not `pass_rate ** k` and
not an estimate of the probability that `k` independent attempts all
succeed; it rewards streaks and penalises alternation. It is defined only for
`n >= k`; otherwise unavailable. Windows never span a gap: if a `run_index`
is missing or excluded as a harness failure, the windows that would contain
it are not counted, and if no complete window remains the task is
unavailable at that `k`. Duplicate `run_index` values for one task are
invalid input and must raise, not be silently accepted. Suite pass^k is the
unweighted mean over eligible tasks, with eligible/total counts.

### Pass rate and mean score

- **Pass rate** = passed gradable trials / gradable trials. Harness failures
  are not in the denominator.
- **Mean score** = mean over gradable trials of the trial's
  `aggregate_score`, which is the mean of its outcomes' scores.

Both are trial-level descriptive numbers. Use them for reading a run; use
pass@k and pass^k with intervals for decisions.

### Bootstrap confidence intervals (suite level)

Percentile bootstrap over **tasks**:

1. Compute the per-task statistic once for every eligible task, in canonical
   order (sorted `task_id`), giving a vector of `T` scores.
2. Draw `B` resamples of size `T` **with replacement**. A task drawn twice
   contributes twice; multiplicity is preserved (issue #44).
3. The suite statistic of each resample is the mean of the drawn scores.
4. The interval is the `alpha/2` and `1 - alpha/2` percentiles of the `B`
   resample statistics, with `alpha = 1 - confidence`.

Rules: `0 < confidence < 1` and `B >= 1`, otherwise `ValueError`. Every
resampling function accepts a `seed`; the same inputs and seed give the same
interval, and reordering the input tasks does not change it. `T = 0` yields
no interval (unavailable). `T = 1` yields a degenerate interval equal to the
single score; present it as "no uncertainty estimable", never as a tight
interval.

`PassAtKAnalyzer.compute_confidence_interval` and
`tracelens.statistics.inference.bootstrap_ci` implement this contract.
pass@k intervals produced before the #44 fix were roughly 20–25 % too
narrow at typical suite sizes: each resample lost about 37 % of its draws to
de-duplication.

### Baseline regression detection

`RegressionDetector` compares, per task and per metric, the mean of the
current trials against the stored baseline mean:

- `delta` and `delta_percent` come from means; a finding below
  `min_delta_percent` (default 5 %) is not reported.
- Significance uses a one-sample t-test against the baseline mean when the
  current sample has `n >= 2` and the baseline has a standard deviation, with
  z-test fallbacks for degenerate cases. A test that cannot be run is
  reported as `insufficient_data`, never as "not significant".
- **Severity is derived from `|delta_percent|` alone** (minor below 5 %,
  moderate 5–15 %, severe above 15 %) and is reported next to significance,
  not combined with it.
- With a `DecisionSpec` on both sides, an absolute delta smaller than the
  noise band (default 0.03 on a 0–1 metric) is marked `within_noise_band` and
  does not block; a changed infrastructure configuration is reported as
  `infra_config_mismatch`.
- Samples are gradable trials only; `TIMEOUT` is included as a failure.

### Run-versus-run comparison (planned, issue #28)

The paired task-level design above is the contract `tracelens compare` must
implement: the per-task statistic under both runs, paired resampling of
tasks, an explicit policy for unmatched or changed tasks, an explicit
estimand (which grader, which metric, which direction is better), and a
three-way verdict: significant change, practically meaningful change against
a stated threshold, or insufficient evidence. Non-significance is not
evidence of equivalence.

## Availability

A number that was not measured is **unavailable**, never zero:

- pass@k or pass^k at a `k` larger than the runs available is `N/A`, with the
  reason and the runs required.
- A suite statistic with zero eligible tasks is `N/A`.
- An interval that could not be estimated is `N/A`, not `[0, 0]`.
- JSON output carries availability explicitly; Markdown, HTML, and the CI
  summary render the same meaning.

Reports show numerator and denominator (eligible tasks / total tasks,
gradable trials / total trials) wherever a subset is summarised, so two runs
are never compared across silently different populations.

## Reproducibility of statistics

- Every resampling or permutation procedure exposes `seed`.
- Canonical iteration order for tasks is sorted `task_id`; for trials within
  a task it is `run_index`. Insertion, completion, or checkpoint-resume order
  never changes a reported number.
- The method, effective sample unit, sample counts, `confidence`, `B`, and
  `seed` are recorded alongside any interval or verdict that is persisted.
- Every run records a `RunProvenance` envelope: per-task content hashes,
  grader identities, runner settings, and the candidate fingerprint. A
  run-versus-run comparison is defined only over runs whose measurement side
  is compatible (`check_compatibility`); tasks are aligned by content, never
  by id alone, and a missing envelope makes compatibility *unknown*, not
  assumed. The baseline gate applies the same rule per task through
  `TaskBaseline.task_hash`. See
  [Run provenance](reproducibility.md#run-provenance).

## Known deviations in the current code

| Behaviour today | Contract says | Resolved by |
|---|---|---|
| pass@k bootstrap de-duplicated repeated task draws and had no seed | multiplicity preserved, seedable, order-independent | #44 (fixed) |
| pass^k used trial insertion (completion) order and could not see gaps | `run_index` order; windows never span gaps; duplicate run indices raise | #45 (fixed) |
| pass@k with `n < k` fell back to the empirical rate `c / n`; pass^k silently dropped such tasks from the suite mean | unavailable, with eligible/total counts | #46 (fixed) |
| Suite pass@k, suite pass^k, and `TrialBatch.pass_rate` counted harness failures as agent failures (all trials in the denominator) | harness failures excluded and reported separately | #46 (fixed) |
| A reliability metric with no eligible task rendered as `0.0` | `N/A` with reason | #46 (fixed) |
| The gate decision was not persisted; a re-rendered report dropped regression data | one gate result across CLI, JSON, Markdown, HTML | #47 (fixed) |
| `compare_metrics` resamples two arms independently | paired task-level resampling for run comparison | #28 |

## Related pages

- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — what each metric answers.
- [Statistical Comparison](statistical-comparison.md) — the inference API.
- [Accuracy Best Practices](accuracy.md) — sample sizes.
- [Reproducibility & DecisionSpec](reproducibility.md) — configuration
  fingerprints.
