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

The gate behind `tracelens run --baseline-check` (issue #111) decides one run. Its inputs are, per task and per metric, the stored
baseline summary (`baseline_value`, `std_deviation`, `sample_size`) and the
current gradable trials, one sample per trial (`TIMEOUT` counts as a
failure). The run spends **one** error budget `alpha = 0.05`. By default the
per-task criterion owns all of it and the suite criterion is reported but
cannot block; `--suite-blocking` lets the suite criterion block too, and the
budget is then split `alpha/2` to each so the run-level total stays `alpha`.
The run is blocked when a live criterion rejects at or above
`--fail-on-regression`. `RegressionDetector` implements the per-task part and
`tracelens.reporting.gate.evaluate_gate` the run-level part.

**Per task.**

- Effect: `delta` = current mean − baseline mean and `delta_percent` relative
  to `|baseline mean|`. A change below `min_delta_percent` (default 5 %) is
  not reported. **Severity is derived from `|delta_percent|` alone** (minor
  below 5 %, moderate 5–15 %, severe from 15 %) and is reported next to the
  evidence, never combined with it.
- Evidence: a one-sided p-value in the observed direction.
    - Whether a metric *is* a 0/1 proportion is decided by **the baseline**
      alone: a current sample can never promote a continuous score to a
      count. A baseline may declare it (`MetricBaseline.is_rate`);
      otherwise it is inferred from the stored summary — the mean is in
      `[0, 1]`, `baseline_value × n_b` is a whole count of successes, and
      any positive recorded spread is one 0/1 data of that size could show.
      (Reading the family off the current values instead made the verdict
      discontinuous in the *upward* direction too: a continuous score of
      five zeros took the exact test and one of five `1e-8`s took a t-test,
      with opposite outcomes.) A declaration cannot supply evidence the
      summary lacks: a mean that is not a whole count over `n_b` is compared
      as a continuous metric whatever the baseline calls it, because the
      exact test would otherwise round it to a count nobody measured.
      Counting the two sides additionally needs current values that *are*
      counts, so a rate whose check produced values off 0 and 1 falls back
      to the continuous test — the data contradicts the baseline and no
      table can be built. **That fallback can change the verdict**, because
      the two tests read the same drop differently; `MetricRegression.test`
      records which one ran, so a rate reported as `welch_t` is a rate whose
      check did not produce counts. A proportion uses **Boschloo's exact
      unconditional test** on the two counts, `round(baseline_value × n_b)`
      of `n_b` against `k_c` of `n_c`, with the two samples as the table's
      **columns**, which is the orientation SciPy's model defines. The test
      is exact: its false-rejection probability is at most the level for
      every true pass rate.
    - A continuous metric uses **Welch's** t-test from the summaries. A
      spread measured as zero on one side is not evidence that the two
      populations share a variance, so the pooled-variance test is not used
      at all — and it is not a measurement of zero variance either, so that
      side does not get a mean known exactly: the spread that *was*
      informative stands for both. Treating the zero literally rejected 38 %
      of unchanged runs with three flat baseline trials against twenty
      scattered current ones, and 91 % against fifty; pooling instead fails
      the other way, which is how a baseline recorded with spread 0 over 100
      trials read `p = 6e-34` against three scattered values. When both
      sides are constant the p-value is the exact permutation value
      `1 / C(n_b + n_c, n_c)`, and when they are constant at the *same*
      value there is no evidence of a difference at all (`p = 1`); a single
      current trial against a measured baseline is a prediction-interval t
      on `n_b - 1` degrees of freedom.
    - **No evidence is invented for the baseline.** Its recorded
      `sample_size` is used as recorded, and a baseline that stored fewer
      than two trials carries no measured spread at all
      (`baseline_n_assumed`): its mean is a declared value, and a
      comparison that has to borrow the current sample's evidence to reach a
      verdict has no valid test (`insufficient_data`). Such a check is
      `undetectable`, which makes the gate unevaluable rather than passing
      or blocking on evidence that was never collected. It is not powerless
      — one stored passing trial against seven straight failures is
      `p = 0.049` on the honest counts, and that blocks — but it cannot
      decide much else. Store baselines from real runs; a one-trial baseline
      needs seven check trials to decide even a total failure, and fifteen
      once two tests share the budget.
- Multiplicity: the p-values are Holm-adjusted across **one family** — every
  compared `(task, metric)` pair, a pair with no finding counting as a test
  that did not reject (`--multiplicity holm`, the default) — so the
  probability that an unchanged suite blocks anywhere is at most the level
  that family is given. Over `m` tests the smallest p-value must reach
  `alpha / m`. One family means one budget: giving each metric its own
  family, as an earlier version did, handed a suite storing both
  `pass_rate` and `mean_score` two independent chances to block, so the
  run-level rate grew with the number of stored metrics instead of staying
  at `alpha`. The price is power, paid where it is actually spent — store
  the metric you gate on. `--multiplicity none` holds every test to `alpha`
  on its own.
- Decision: a task blocks when a finding is significant (adjusted p-value at
  or below `alpha`), not `within_noise_band`, and at or above the severity
  threshold. Every change above the floor is reported with its evidence. A
  drop that is not significant is `underpowered` and carries
  `trials_needed`: the check size at which the observed rates would decide
  it, or, when the stored baseline is too small for any check to decide it,
  the size on each side (`trials_needed_on_both_sides`). A 0/1 comparison
  whose sizes cannot reject even a total failure is `undetectable`.
- With a `DecisionSpec` on both sides, an absolute delta smaller than the
  noise band (default 0.03 on a 0–1 metric) is marked `within_noise_band` and
  does not block; a changed infrastructure configuration is reported as
  `infra_config_mismatch`.

**Suite.** For each metric with two or more checked tasks: `d_t` = current
mean − baseline mean per task, the mean over tasks, and the task bootstrap
interval and sign-flip p-value of the run-versus-run section below, one-sided
in the regression direction (half the two-sided value; exact when both sides
of every task have the same trial count, approximate otherwise). One
criterion is formed per metric and they share the suite level through the
same Holm adjustment the per-task family uses, so that half of the budget is
spent once rather than once per stored metric. It sees a broad regression
that no single task can show and does not react to one task among many.

It is **reported but not blocking by default.** Sign-flipping is valid only
when the per-task differences are independent under the null, and task
outcomes in one run frequently are not: one shared infrastructure wobble,
one shared sampling seed, one bad deploy moves many tasks together. Under
dependence the null distribution is too narrow and the p-value is
anti-conservative. Simulating an unchanged agent over 2000 runs of a suite
with 40 deterministic tasks and 10 flaky ones at `p = 0.8`, five trials a
side: with the flaky tasks independent the suite criterion blocked 0.0 % of
runs; with them sharing one run-level outcome it blocked 11.6 %. TraceLens
therefore does not let an unvalidated level gate a merge. `--suite-blocking`
switches it on for suites whose task outcomes are known to be independent,
and then each criterion is held to `alpha/2`.

**Unevaluable.** A check that could not have blocked authorizes nothing:
when every checked task is `undetectable`, the gate is unevaluable and names
the trials per task it would need. The suite criterion can rescue such a
check only when it is allowed to block at all and enough tasks could move
together: its sign-flip p-value is at best `2^-T`, and the criteria share
their half of the budget through the same Holm adjustment, so the best any
one of them can reach is `m × 2^-T` over `m` criteria. That has to meet
`alpha/2`, the level suite blocking leaves: six tasks for a suite storing
one metric (not five), seven for two. Bounding the raw `2^-T` instead
declared a check evaluable that no test could have rejected — six tasks
storing two metrics each, every one collapsing from 4/4 to 0/4, reported
`passed`. When only some tasks are undetectable the gate decides on the
others and warns.

**Error rates.** Exact enumeration over both binomial samples with the real
detector (`scripts/gate_error_rates.py` prints the full tables; `T` is the
size of the Holm family — the number of compared `(task, metric)` pairs —
and the per-test level is `alpha / T`). These are the whole run-level
rates, because the suite criterion does not block by default.

Probability that one unchanged task with true pass rate `p` blocks by chance:

| p | baseline n | check n | T=1 | T=2 | T=10 | T=50 |
|---|---|---|---|---|---|---|
| 0.8 | 5 | 5 | 1.9 % | 0.2 % | 0.0 % | 0.0 % |
| 0.8 | 20 | 5 | 4.2 % | 1.8 % | 0.4 % | 0.1 % |
| 0.8 | 20 | 20 | 3.7 % | 2.2 % | 0.5 % | 0.1 % |
| 0.5 | 5 | 5 | 3.0 % | 1.1 % | 0.1 % | 0.1 % |
| 0.5 | 20 | 20 | 4.1 % | 2.0 % | 0.4 % | 0.1 % |

Deterministic tasks add nothing; a suite with `F` flaky tasks blocks by
chance on some task with probability `1 − (1 − q)^F`. Without the correction
(`T=1` column, `--multiplicity none`) ten flaky tasks at `p = 0.8` with five
trials a side give 17.6 % false alarms per run and fifty give 62 %; under
Holm over fifty tests the same suites give 0.1 % and 0.5 %. These are the
whole run-level rate, because the suite criterion does not block by default.
With `--suite-blocking` each criterion is held to `alpha/2`, and the suite
criterion's own level holds only under the independence assumption stated
above.

Probability that one regressed task blocks while the others are unchanged:

| drop | baseline n | check n | T=1 | T=2 | T=10 | T=50 |
|---|---|---|---|---|---|---|
| 1.0 → 0.4 | 5 | 5 | 68.3 % | 33.7 % | 7.8 % | 7.8 % |
| 1.0 → 0.4 | 10 | 5 | 91.3 % | 68.3 % | 33.7 % | 7.8 % |
| 1.0 → 0.4 | 20 | 5 | 91.3 % | 91.3 % | 68.3 % | 33.7 % |
| 1.0 → 0.4 | 10 | 10 | 94.5 % | 94.5 % | 63.3 % | 38.2 % |
| 1.0 → 0.4 | 20 | 20 | 100 % | 100 % | 99.8 % | 97.9 % |
| 1.0 → 0.6 | 5 | 5 | 31.7 % | 8.7 % | 1.0 % | 1.0 % |
| 1.0 → 0.6 | 20 | 5 | 66.3 % | 66.3 % | 31.7 % | 8.7 % |
| 1.0 → 0.6 | 20 | 20 | 98.4 % | 94.9 % | 87.4 % | 58.4 % |
| 1.0 → 0.0 | any | 5 | 100 % | 100 % | 100 % | 100 % |

Check trials needed to decide a total failure after a perfect baseline:

| baseline n | T=1 | T=2 | T=10 | T=50 |
|---|---|---|---|---|
| 1 | 7 | 15 | >60 | >60 |
| 2 | 3 | 4 | 10 | 23 |
| 5 | 2 | 2 | 4 | 5 |
| 10 | 1 | 2 | 2 | 3 |
| 20 | 1 | 1 | 2 | 3 |

What this means in practice: the baseline is the long-lived side, so store
it from ten runs or more; five-trial checks then decide a 60-point drop on a
single task in a small suite most of the time, and any total failure. A
baseline of one stored trial can decide almost nothing, and says so rather
than borrowing the check's evidence to look decisive. In a fifty-test family
with five trials a side, only a total failure of one task is decidable per
task; a broad drop across many tasks is what the suite statistic reports,
though acting on it is a decision the maintainer opts into. The `trials_needed` note on every underpowered finding says
what would decide the drop that was actually observed. A regression that
the run reports as not significant is a reason to rerun with more trials,
not a clean pass.

### Run-versus-run comparison (`tracelens compare`, issue #28)

`tracelens compare BASELINE-trials.json CANDIDATE-trials.json` decides whether
a candidate run is better, worse, or indistinguishable from a baseline run of
the same eval set. The estimand and the sampling unit are fixed here; the
command implements them and records them in its output.

**Inputs.** Two `--save-trials` artifacts. Aggregate results files do not
contain per-trial samples and are rejected with a message naming the required
input. Each artifact's provenance decides comparability
([Run provenance](reproducibility.md#run-provenance)):

- `incompatible` (task content changed, tasks added or removed, or different
  graders) makes the comparison **unevaluable** (exit 2). The default never
  drops unmatched tasks silently; `--unmatched-tasks exclude` compares the
  shared, unchanged tasks and reports the excluded ones by id and count. A
  grader difference is never overridden: a different ruler is a different
  measurement.
- `unknown` (an artifact without provenance) aligns tasks by id only. The
  output labels the comparison as such; `--require-provenance` makes it
  unevaluable instead.
- The candidate side of the provenance (adapter identity, `DecisionSpec`
  diff) is printed as "what changed" next to "what moved". It supports
  attribution, not proof of cause.

**Estimand.** One metric with one direction, chosen explicitly:

- `pass_rate` (default; higher is better): a trial's value is 1 if it
  passed, else 0.
- `mean_score` (higher is better): a trial's value is its
  `aggregate_score`.
- `<grader_id>.<metric_name>`: the named outcome metric, with
  `--direction higher|lower` stating which way is better (a latency budget
  metric, for example, is `lower`).

With several graders, `pass_rate` and `mean_score` follow the trial-level
rule (all graders passed; mean of grader scores); `--grader ID` restricts
both to that grader's outcome. Direction is normalised so that a positive
effect is always an improvement.

**Trial validity.** Only gradable trials contribute (`Trial.is_gradable`:
`COMPLETED`, `FAILED`, or `TIMEOUT` without a grader crash). Infra errors,
grader crashes, and never-run trials are excluded and counted per run; for
`pass_rate` a `TIMEOUT` counts as a failure, as the report does. A trial with
no value for the selected metric (a missing outcome metric, or no score) is
excluded and counted. Unavailable evidence is never a zero delta.

**Sampling unit and statistic.** The unit is the task, matched across runs:

1. For each shared task `t` and each run, the task statistic `θ_A(t)` /
   `θ_B(t)` is the mean of the trial values of that task in that run.
   Repeated trials of one task are averaged into it; they are not
   independent samples of the suite, and equal `run_index` values do not pair
   trials across runs.
2. The paired difference is `d_t = θ_B(t) − θ_A(t)`, direction-normalised.
3. The effect is `Δ = mean_t d_t` over the `T` shared tasks with a value on
   both sides. It equals the difference of the two suite means over the same
   task set, so heterogeneous task difficulty cancels instead of widening the
   interval.
4. The interval is a percentile bootstrap over the `T` paired differences:
   `B` resamples of size `T` with replacement, multiplicity preserved, and
   the `alpha/2` and `1 − alpha/2` percentiles of the resample means.
   `confidence`, `B`, and `seed` are inputs; the same inputs and seed
   reproduce the result exactly, and task order never matters.
5. The p-value is a paired sign-flip permutation test: under the null of no
   within-task difference, each `d_t` is equally likely to carry either sign,
   and the two-sided p-value is the fraction of `B` random sign assignments
   (counting the observed one) whose mean is at least as extreme as `|Δ|`.
   The assignments are drawn with the same `seed`; when `T ≤ 12` and
   `2^T ≤ B` all assignments are enumerated instead and the p-value is
   exact.

**Verdict.** Given the practical threshold `τ` (`--threshold`, an absolute
delta on the metric's scale; default 0.03), the level `α = 1 − confidence`,
the interval `[lo, hi]`, and the sign-flip p-value `p`, two rules come before
the table (issue #112):

- **Evidence floor.** With `T` paired tasks the exact sign-flip p-value
  cannot fall below `2 / 2^T`. It gets there only when every difference is
  non-zero and all share one sign, so that no sign assignment but the
  observed one and its mirror image is as extreme. A sampled p-value also
  cannot fall below `1 / (B + 1)`. When that floor is above `α`, no difference can be
  significant however large it is, so there is no verdict: fewer than 6
  tasks at 0.95, 5 at 0.90, 8 at 0.99. The output names the p-value the
  test cannot get below and the tasks a verdict needs.
- **Significance needs agreement.** A difference is significant when the
  interval excludes 0 *and* `p ≤ α`. On few tasks the percentile interval is
  too narrow: on its own it called a regression in 21 % of no-change
  comparisons at two tasks and 6.7 % at six, beside a printed `p = 0.5`. The
  sign-flip test is exact under the null of no change, so requiring both
  keeps the regression rate at or below the level and the verdict never
  contradicts the printed p-value.

| Evidence | Verdict | Exit |
|---|---|---|
| no task has a value on both sides, `T < 2`, or the smallest attainable `p` is above `α` | insufficient evidence | 2 |
| significant and `Δ ≤ −τ` | regression | 1 |
| significant, `Δ ≥ τ`, and `lo > −τ` | improvement | 0 |
| significant, `−τ < Δ < τ`, and `lo > −τ` | significant but below the practical threshold | 0 |
| not significant, and the interval lies inside `(−τ, τ)` | equivalent within the threshold | 0 |
| anything else: not significant with the interval reaching `−τ` or `τ`, or significant with `Δ > −τ` but `lo ≤ −τ` | inconclusive: more runs or tasks needed | 2 |

Significance (the interval and the p-value agree), practical relevance
(`|Δ|` against `τ`), and evidence (the interval's extent against `τ`) are
three separate readings, and the output reports all three. Non-significance
is never equivalence: only an interval inside `(−τ, τ)` supports "no
meaningful change". Likewise a significant change below the threshold passes
only when the interval also stays above `−τ`. Every verdict that exits 0
therefore rules out a regression of `τ` or more (`lo > −τ`): while the
interval reaches `−τ`, narrowing it toward harm can move the verdict from 2
to 1, never to 0.
Exit codes follow the CLI contract (0 success, 1 negative result, 2
unevaluable). `--observe` makes every *evaluated* comparison exit 0, for
dashboards and exploratory runs; incompatible, empty, or aggregate-only inputs
still exit 2.

**Small suites.** The floor is a property of the task count, not of how
decisive each task looks. Two tasks that both went from always passing to
always failing give `p = 0.5` and no verdict, and so do two tasks that did
not move at all: "equivalent" is a claim that needs the same evidence as
"regression". A task whose difference is exactly 0 carries no sign, so it
adds nothing to the test: six tasks of which five collapsed and one did not
move give `p = 4/64` and an inconclusive verdict. Above the floor the rule
trades power for calibration on small suites, as the table shows. The
cheapest way to more power is more tasks; more trials per task help when the
per-task differences are noisy.

**Error rates.** How often the verdict is a regression, out of 2000 simulated
comparisons per cell (threshold 0.03, confidence 0.95, B = 10000;
`scripts/compare_error_rates.py` regenerates this table). *No change* draws
each task's paired difference from a normal distribution around 0 with
standard deviation 0.1, and *drop of 0.10* around -0.10; the pass/fail columns
observe each task 5 times a side at a pass probability drawn from U(0.1, 0.9),
which the drop lowers by 0.20. *Interval alone* is the verdict before #112.
Below 6 tasks there is no verdict. Near 2.5 % a cell's sampling error is about
0.7 percentage points either way (two standard errors).

| tasks T | no change, interval alone | no change | drop of 0.10 | pass/fail, no change | pass/fail, 20-point drop |
|---|---|---|---|---|---|
| 2 | 21.1 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| 3 | 11.6 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| 4 | 9.3 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| 5 | 7.3 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| 6 | 6.7 % | 1.8 % | 36.9 % | 0.1 % | 6.2 % |
| 8 | 5.7 % | 2.4 % | 67.2 % | 0.4 % | 24.2 % |
| 10 | 5.5 % | 2.5 % | 80.0 % | 1.3 % | 39.1 % |
| 15 | 4.9 % | 3.3 % | 94.5 % | 1.0 % | 65.5 % |
| 20 | 2.7 % | 2.2 % | 99.1 % | 1.6 % | 82.6 % |
| 30 | 3.1 % | 2.5 % | 100.0 % | 1.7 % | 94.6 % |

**Output.** The terminal summary and the `--output` JSON carry the same
fields: the method (`paired task bootstrap`), the unit, the metric and its
direction, the grader selection, per-run trial counts (gradable, and excluded
by reason), task counts (shared, and excluded by reason), `Δ` (with the raw
candidate-minus-baseline delta for lower-is-better metrics), `[lo, hi]`,
`confidence`, `B`, `seed`, the p-value, the smallest p-value the test could
return with these tasks and draws (`min_attainable_p`) and the fewest tasks
a verdict needs at this confidence (`min_tasks`), `τ`, the readings
(`significant`, which requires agreement; `interval_excludes_zero`, the
interval alone; `meaningful`), the verdict, the exit code, the per-task `d_t`
with each side's trial count (largest movers first), the compatibility
report, and the candidate diff.

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
| No run-versus-run command; `compare_metrics` resampled two arms independently | `tracelens compare` per the contract above: paired task-level resampling, explicit estimand, three-way verdict | #28 (fixed) |
| `tracelens compare` decided on the percentile interval alone and never consulted the sign-flip p-value it printed; with few tasks the interval is too narrow, so a no-change comparison was called a regression 21 % of the time at two tasks and 6.7 % at six, beside `p = 0.5` | significance needs the interval and the p-value to agree, and there is no verdict below the tasks at which the test can reach the level (6 at 0.95) | #112 (fixed) |
| A significant change smaller than `τ` passed even when the interval reached past `−τ`, so making a regression more certain could move exit 2 to exit 0 | "below the threshold" requires `lo > −τ`; every verdict that exits 0 rules out a regression of `τ` or more | #112 (fixed) |
| The gate's fallback for a zero-variance baseline divided the delta by the sample SD, not the standard error, so a drop's p-value never tightened with `n` (1.0 → 0.4 over 5, 10, or 100 trials all read p ≈ 0.22) | exact test on the two counts for 0/1 metrics; SE-based t-tests otherwise | #111 (fixed) |
| A drop that was not significant was dropped from stdout, JSON, Markdown, and HTML, so an underpowered check read like a clean pass | every change above the floor is reported with its evidence, `underpowered`, and `trials_needed` | #111 (fixed) |
| The baseline's `sample_size` was never read; the stored mean was treated as exact | two-sample tests on both sample sizes, used as recorded | #111 (fixed) |
| The run blocked when any task blocked, with no multiplicity control and no suite-level criterion; a check that could not have blocked passed | one Holm family over every compared (task, metric) pair, a reported suite-level statistic, and unevaluable when nothing could have blocked | #111 (fixed) |
| Boschloo's table was built with the two samples as rows, which fixes the wrong margin; at unequal sizes it changed the verdict (7/7 vs 2/4 read 0.0420 instead of 0.0538) | the two samples are the table's columns, as SciPy's model defines | #111 (fixed) |
| A baseline with fewer than two stored trials was credited with as many trials as the check, and its missing spread was taken from the current sample | recorded sizes are used as recorded; a baseline with no measured spread yields no test, and the check is unevaluable | #111 (fixed) |
| A spread measured as zero on one side switched the comparison to a pooled-variance t-test, assuming a shared population variance nothing showed | Welch throughout; the pooled test is not used | #111 (fixed) |
| Whether a metric was a proportion was read off the current sample's endpoints, so an infinitesimal change flipped the test family and the verdict | the baseline decides, by declaration or from its stored summary | #111 (fixed) |
| Each metric received its own Holm family and the suite criterion its own `alpha`, so the run-level rate grew with the number of stored metrics | one family, one budget; the suite criterion reports by default and splits the budget when switched on | #111 (fixed) |

## Related pages

- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — what each metric answers.
- [Statistical Comparison](statistical-comparison.md) — the inference API.
- [Accuracy Best Practices](accuracy.md) — sample sizes.
- [Reproducibility & DecisionSpec](reproducibility.md) — configuration
  fingerprints.
