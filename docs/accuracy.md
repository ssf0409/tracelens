# Evaluation Accuracy Best Practices

You ran an eval and got a number. **Can you trust it?** This page is about the
difference between "we ran an eval" and "we trust this eval" — sample sizes,
grader calibration, and the pitfalls that produce confident-but-wrong signals.

## The Accuracy Problem

Agent evaluations suffer from three failure modes:

1. **False positives** — agent passes but shouldn't (grader too lenient).
2. **False negatives** — agent fails but should pass (grader too strict or flaky).
3. **Non-determinism noise** — same input produces different results across runs.

A bad evaluation is worse than no evaluation — it builds false confidence or
blocks valid work.

## Sample Size Guidelines

The single most common accuracy mistake is too few samples. How many runs and
tasks you need depends on the scenario:

### Runs Per Task

| Scenario | Minimum Runs | Recommended |
|----------|-------------|-------------|
| Deterministic agent + CodeGrader | 1 | 1 |
| Non-deterministic agent | 3 | 5–10 |
| LLMGrader (any agent) | 3 | 5 |
| High-stakes decision | 10 | 20+ |

**Rule of thumb:** if your confidence-interval width is `> 0.1`, you need more
runs. Put a CI on every metric with `estimate_metric` and watch `ci_width` — the
mechanics are in [Statistical Comparison](statistical-comparison.md).

### Tasks in Eval Set

| Eval Set Maturity | Minimum Tasks |
|-------------------|---------------|
| Initial (prototype) | 10 |
| Development | 20–50 |
| Production CI | 50+ |

Start with real failure cases from your agent's history, not synthetic tasks.

## Grader Calibration

A grader is only as trustworthy as its agreement with reality. Deterministic and
LLM graders fail differently.

### CodeGrader best practices

CodeGraders are deterministic and reproducible. Key design considerations:

1. **Metric design** — choose metrics that capture what matters, not what's easy
   to measure.
2. **Threshold selection** — set pass thresholds from observed performance, not
   guesses.
3. **Boundary testing** — test grader behavior at edge cases (empty output,
   `None`, extreme numbers). Grade defensively:

```python
class FinancialGrader(CodeGrader):
    def compute_metrics(self, transcript, task):
        output = transcript.final_output
        # Defensive: handle missing or malformed output
        if not output or "returns" not in output:
            return {"sharpe_ratio": 0.0, "max_drawdown": -1.0}
        ...
```

### LLMGrader best practices

LLM judges are non-deterministic and prone to drift. Mitigate with:

1. **Deterministic decoding** — judge at temperature 0 to reduce variability.
2. **Structured output** — request JSON with specific fields, not free-form text.
3. **Rubric anchoring** — define what each score level means concretely.
4. **Boundary examples** — include 1–2 "definitely pass" and "definitely fail"
   examples in the prompt.
5. **Score normalization** — map LLM scores (1–10) to 0–1 consistently.

```python
def build_grading_prompt(self, transcript, task):
    return """Score this output 1-10 on completeness.

    SCORING RUBRIC:
    - 1-3: Missing major required elements
    - 4-6: Addresses task partially, notable gaps
    - 7-8: Addresses all requirements adequately
    - 9-10: Exceeds requirements with exceptional detail

    EXAMPLES:
    - Score 2: Output says "I don't know" (missing all elements)
    - Score 9: Detailed plan with specific actions, timelines, resources

    Return ONLY: {"score": N, "feedback": "brief explanation"}"""
```

### Gate vs. score in a CompositeGrader

When you combine graders, decide which ones are hard gates and which contribute
to the score (set via `GraderConfig`'s `EvalPolicy`):

| Grader purpose | Policy | Behavior |
|---------------|--------|----------|
| Safety / format validation | `GATE` | Any violation fails the trial |
| Quality / style scoring | `TRACK` (or `WARN`) | Contributes to the weighted score |

Use `GATE` sparingly — it creates binary signals. See the
[Grader Library](grader-library.md) for the built-in graders and their default
policies, and [Evaluating a Real Agent §4](real-agent.md) for the gate-plus-judge
pattern.

### Calibrating an LLM judge against humans

LLM graders drift; the only way to know is to compare them against human grades
on a periodic sample. TraceLens owns the selection and the agreement math
(`tracelens sample` / `reconcile`); you bring the human grades. The full
workflow, strategies, and the correlation thresholds to watch are in the
**[Human-Eval Calibration](human-eval.md)** guide, with a runnable, API-key-free
demo in
[`examples/human_eval_calibration.py`](https://github.com/ssf0409/tracelens/blob/main/examples/human_eval_calibration.py).

## Handling Non-Determinism

Set run counts to the scenario (3 minimum for non-deterministic, 5 for CI, 10
for benchmarking), and **always report both statistics**:

```python
gen = ReportGenerator(k_values=[1, 3, 5], consistency_k_values=[2, 3, 5])
report = gen.build_report(batch)
```

A high pass@k with a low pass^k means the agent is *capable but inconsistent* —
a different problem than an agent that simply can't solve the task. See
[pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md).

To make sure you're comparing like with like across runs, stamp each run with a
`DecisionSpec` so a difference is attributable to the agent/prompt/model rather
than the environment — see [Reproducibility & DecisionSpec](reproducibility.md).

## Baselines: setting thresholds from variance

Set regression thresholds from observed variance, not arbitrary numbers:

1. Run your eval suite 5–10 times on the same agent version.
2. Compute the standard deviation for each metric.
3. Set the threshold at **2× standard deviation** — this catches real
   regressions while tolerating natural variance.

```python
# If observed std of pass_rate is 0.03, ~2x std ≈ 6%:
baseline.add_metric(metric_name="pass_rate", value=0.85, std=0.03, relative_threshold=0.06)
```

Storing, promoting, and comparing baselines (including canary baselines for
safety-critical metrics that must never regress) is covered end to end in the
[Baseline Regression Tutorial](baseline-regression-tutorial.md).

## Common Pitfalls

### 1. Grading paths, not outcomes

**Wrong:** check that the agent used specific tools in a specific order.
**Right:** check that the final output meets requirements, regardless of how it
got there.

### 2. Too few samples

Running 1 trial per task gives a binary signal with no statistical power. Run at
least 3.

### 3. Stale baselines

Baselines established months ago may not reflect current expectations. Review
stale baselines regularly rather than trusting them indefinitely.

### 4. LLM grader drift

LLM grader behavior changes when the underlying model is updated. After any model
upgrade: re-run calibration with human scores, compare old vs new grading on the
same transcripts, and update prompts if correlation drops below `0.7`.

### 5. Ignoring pass^k

High pass@k can mask reliability problems. A task with pass@5 = 0.99 but
pass^5 = 0.20 means the agent almost always succeeds *eventually* but fails 80%
of the time when you need 5 consecutive successes. For production use, pass^k
often matters more.

### 6. Overfitting the eval suite

If you only add tasks where the agent fails, the suite becomes a regression test,
not a capability test. Regularly add new tasks from fresh failure cases and
retire tasks that have passed consistently for months.

## Evaluation Maturity Model

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **1 — Manual** | Ad-hoc spot checking | No automation, no baselines |
| **2 — Basic** | Automated eval suite | CodeGrader, num_runs=1, CI output |
| **3 — Statistical** | Non-determinism handled | num_runs ≥ 3, pass@k + pass^k, baselines |
| **4 — Calibrated** | Human-validated grading | Weekly calibration, LLMGrader correlation > 0.7 |
| **5 — Production** | Full pipeline with dashboard | HTML dashboard, regression gating, DecisionSpec tracking, canary baselines |

Most teams should aim for Level 3 initially and progress to Level 4–5 as their
agent matures.
