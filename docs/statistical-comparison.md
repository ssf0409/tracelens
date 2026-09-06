# Statistical Comparison

A single pass rate means nothing on its own. If version A scores `0.78` and version B scores `0.80`, is B better — or did you just get luckier dice on B? You cannot answer that from the two numbers alone. You need the *spread*: how much would these numbers move if you reran the eval?

TraceLens gives you three tools for this, all in `tracelens.statistics.inference`:

1. **Estimate** — put a confidence interval on one metric (`estimate_metric` / `bootstrap_ci`).
2. **Compare** — put a confidence interval on the *difference* between two sets, plus an effect size and a p-value (`compare_metrics`).
3. **pass@k / pass^k estimators** — capability and reliability over repeated trials (`pass_at_k`, `pass_to_k`); see [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) for the intuition.

This page is the depth reference for the first two. For the applied "did my new version regress?" workflow, see [Comparing Versions](comparing-versions.md). For sample-size and CI-width rules of thumb, see [Accuracy Best Practices](accuracy.md).

The estimate/compare functions are exported from the package root: `from tracelens import estimate_metric, compare_metrics, bootstrap_ci, MetricEstimate, ComparisonResult`.

---

## 1. Estimating one metric

`estimate_metric(values, confidence=0.95, n_bootstrap=10000, seed=None)` turns a list of per-trial scores into a point estimate with a bootstrap confidence interval. It returns a `MetricEstimate`.

```python
from tracelens import estimate_metric

# Per-trial pass/fail (or any 0–1 metric) from one eval run.
scores = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]  # 8/10 = 0.80
est = estimate_metric(scores, confidence=0.95, seed=0)

print(est)                       # MetricEstimate(0.8000 [0.5000, 1.0000], n=10)
print(est.mean, est.ci_width)    # 0.8 0.5
```

A 95% CI of `[0.50, 1.00]` on a pass rate of `0.80` is the honest message here: **with only 10 trials, "0.80" could plausibly be anything from a coin flip to perfect.** That width is your signal to run more trials.

`MetricEstimate` carries:

| Field / property | Meaning |
|---|---|
| `mean` | Point estimate (sample mean) |
| `std` | Sample standard deviation (`ddof=1`; `0.0` when `n <= 1`) |
| `n` | Number of samples |
| `ci_lower`, `ci_upper` | Bootstrap CI bounds (`None` until computed) |
| `confidence` | Confidence level used (default `0.95`) |
| `se` (property) | Standard error of the mean; `inf` when `n <= 1` |
| `ci_width` (property) | `ci_upper - ci_lower`, or `None` if bounds are missing |

If you only need the raw bounds, `bootstrap_ci(values, confidence=0.95, n_bootstrap=10000, statistic="mean", seed=None)` returns a `(point_estimate, lower_bound, upper_bound)` tuple. `statistic` accepts `"mean"`, `"median"`, or `"std"`.

```python
from tracelens import bootstrap_ci

point, lo, hi = bootstrap_ci(scores, statistic="mean", seed=0)
# (0.8, 0.5, 1.0)
```

---

## 2. Comparing two sets

`compare_metrics` is the workhorse. It takes the baseline and current sample lists and returns a `ComparisonResult` with everything you need to judge a delta.

```python
from tracelens import compare_metrics

baseline = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]   # v1: 0.80
current  = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]   # v2: 0.90

result = compare_metrics(baseline, current, compute_p_value=True, seed=0)

print(result.delta)            # 0.1    (current - baseline)
print(result.relative_delta)   # 0.125  (delta / |baseline mean|)
print(result.ci_lower, result.ci_upper)  # -0.2 0.4  (CI of the difference)
print(result.is_significant)   # False  — CI of the difference includes 0
print(result.cohens_d)         # 0.268  — small effect
print(result.p_value)          # 1.0    — not significant
```

Full signature:

```python
compare_metrics(
    baseline_values, current_values,
    confidence=0.95, n_bootstrap=10000,
    compute_p_value=False, seed=None,
) -> ComparisonResult
```

`ComparisonResult` fields:

| Field | Meaning |
|---|---|
| `baseline`, `current` | The two `MetricEstimate`s |
| `delta` | `current.mean - baseline.mean` |
| `relative_delta` | `delta / abs(baseline.mean)` (`inf` if baseline mean is 0 and delta isn't) |
| `ci_lower`, `ci_upper` | Bootstrap CI of the **difference** |
| `confidence` | Confidence level used |
| `is_significant` | `True` iff the difference CI **excludes 0** |
| `cohens_d` | Effect size (see below) |
| `p_value` | Permutation-test p-value, or `None` unless `compute_p_value=True` |

Convenience properties: `is_regression` (significant **and** `delta < 0`), `is_improvement` (significant **and** `delta > 0`), and `effect_magnitude` (a string band derived from `cohens_d`).

### How to read significance

The rule is mechanical: **the difference is significant when its confidence interval excludes 0.** TraceLens computes this directly — `is_significant = not (ci_lower <= 0 <= ci_upper)`.

In the example above, `0.90` vs `0.80` *looks* like a clear win, but the difference CI is `[-0.2, 0.4]` — it straddles 0, so `is_significant` is `False`. Ten trials each is not enough to distinguish a 10-point gap from noise. Run more before you celebrate (or block CI). Contrast this with the [Comparing Versions](comparing-versions.md) example, where 60 samples per side produce a CI of `[+0.119, +0.201]` that clearly excludes 0.

### Summary-only comparison

If you've discarded raw trials and only kept summary stats, use `compare_to_baseline_summary(baseline_mean, baseline_std, baseline_n, current_mean, current_std, current_n, confidence=0.95)`. It returns the same `ComparisonResult`, but the CI comes from a Welch's t-test approximation rather than the bootstrap. Prefer `compare_metrics` with raw values whenever you have them.

### Two runs of the same eval set: `compare_runs`

`compare_metrics` resamples its two lists independently, so it is the wrong
tool for two runs of the *same tasks*: the values are paired by task, repeated
trials of a task are not independent samples, and between-task difficulty would
be counted as noise. For that case use `compare_runs(baseline_batch,
candidate_batch, metric=..., threshold=...)`, the function behind
`tracelens compare`. It aligns tasks by content through the runs' provenance,
pairs each task's statistic, bootstraps over tasks, adds a sign-flip p-value,
and returns a `RunComparison` with a verdict and exit code. The full definition
is in the [statistical contract](statistical-contract.md#run-versus-run-comparison-tracelens-compare-issue-28);
the walkthrough is [Comparing Versions](comparing-versions.md).

---

## 3. Effect size

Significance and importance are different questions. A tiny, uninteresting difference becomes "significant" if you throw enough samples at it. **Cohen's d** answers the second question: how big is the gap *relative to the noise*?

`cohens_d(baseline_values, current_values) -> float` returns the standardized mean difference using pooled standard deviation (positive means current > baseline). It returns `0.0` if either sample has fewer than 2 values or the pooled std is 0.

```python
from tracelens.statistics.inference import cohens_d  # not re-exported at root

d = cohens_d(baseline, current)  # 0.268 for the data above — a small effect
```

Rough bands (matching `ComparisonResult.effect_magnitude`):

| `|d|` | Magnitude |
|---|---|
| `< 0.2` | negligible |
| `0.2 – 0.5` | small |
| `0.5 – 0.8` | medium |
| `>= 0.8` | large |

Read it alongside significance: a **significant + large** delta is a real, meaningful change; a **significant + negligible** delta is real but probably not worth acting on; a **non-significant** delta is "we can't tell yet, get more data."

> `cohens_d` is not re-exported from the package root — import it from `tracelens.statistics.inference`, or just read `result.cohens_d`, which is always populated by `compare_metrics`.

---

## 4. Permutation test

`permutation_test(baseline_values, current_values, n_permutations=10000, alternative="two-sided", seed=None) -> float` returns a p-value. It pools the two samples, repeatedly reshuffles the labels, and asks: *how often does a random split produce a mean difference at least as extreme as the one we actually observed?* `alternative` accepts `"two-sided"`, `"greater"`, or `"less"`.

```python
from tracelens.statistics.inference import permutation_test

p = permutation_test(baseline, current, seed=0)  # 1.0 here — far from significant
```

This is the path `compare_metrics(..., compute_p_value=True)` runs internally (using `n_bootstrap` as the permutation count), populating `ComparisonResult.p_value`.

**What the p-value means:** the probability of seeing a difference this large *if the two versions were actually identical*. Small p (e.g. `< 0.05`) ⇒ unlikely to be pure chance.

**Its limits:** a p-value tells you *whether* there's a difference, not *how big* or *whether it matters* — that's what Cohen's d is for. It's also non-deterministic without a fixed `seed`, and with few samples even a real difference can post a large p-value. Treat the difference CI as your primary signal and the p-value as a cross-check; don't chase the `0.05` threshold.

---

## 5. How many samples?

Every tool above gets sharper with more data. The lever is `num_runs` (and the number of tasks):

- **More `num_runs` per task tightens every CI.** The `[0.50, 1.00]` interval from 10 trials shrinks fast as you add runs — and a tighter difference CI is what flips a real improvement from "not significant" to "significant."
- **Rule of thumb from [Accuracy Best Practices](accuracy.md): if a CI width exceeds `0.1`, you need more runs or more tasks.** That doc has a full sample-size table.
- **The pass@k / pass^k estimators need enough trials per task.** `pass_at_k_estimator(results_per_task, k)` and `pass_to_k_estimator(results_per_task, k)` only compute on tasks that have at least `k` samples; tasks with too few are skipped (`pass^k`) or fall back to an empirical rate (`pass@k`). With `k=5` and only 3 runs per task, you're not measuring what you think you are.

A practical loop: run, call `estimate_metric`, check `ci_width`; if it's above `0.1`, raise `num_runs` and rerun. Once the CIs are tight, `compare_metrics` between versions becomes trustworthy.

---

## See also

- [Comparing Versions](comparing-versions.md) — the applied A/B workflow that uses these functions.
- [Accuracy Best Practices](accuracy.md) — sample-size table and CI-width thresholds.
- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — capability vs reliability intuition for the estimators.
- [API Reference](reference.md) — full public API listing.
