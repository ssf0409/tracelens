# Evaluation Accuracy Best Practices

How to build evaluations that produce reliable, actionable signals.

## The Accuracy Problem

Agent evaluations suffer from three failure modes:

1. **False positives** — agent passes but shouldn't (grader too lenient)
2. **False negatives** — agent fails but should pass (grader too strict or flaky)
3. **Non-determinism noise** — same input produces different results across runs

A bad evaluation is worse than no evaluation — it builds false confidence or blocks valid work.

## Sample Size Guidelines

### Runs Per Task

| Scenario | Minimum Runs | Recommended |
|----------|-------------|-------------|
| Deterministic agent + CodeGrader | 1 | 1 |
| Non-deterministic agent | 3 | 5-10 |
| LLMGrader (any agent) | 3 | 5 |
| High-stakes decision | 10 | 20+ |

**Rule of thumb**: If your confidence interval width is > 0.1, you need more runs.

### Tasks in Eval Set

| Eval Set Maturity | Minimum Tasks |
|-------------------|---------------|
| Initial (prototype) | 10 |
| Development | 20-50 |
| Production CI | 50+ |

Start with real failure cases from your agent's history, not synthetic tasks.

### Bootstrap CI as Signal

Use `PassAtKAnalyzer.analyze_with_ci()` to get confidence intervals:

```python
analyzer = PassAtKAnalyzer(k_values=[1, 5])
results = analyzer.analyze_with_ci(pass_results, confidence=0.95)
# {"pass@5": {"value": 0.92, "lower": 0.85, "upper": 0.97}}

ci_width = results["pass@5"]["upper"] - results["pass@5"]["lower"]
if ci_width > 0.1:
    print("Warning: wide CI — add more tasks or runs")
```

## Grader Calibration

### CodeGrader Best Practices

CodeGraders are deterministic and reproducible. Key design considerations:

1. **Metric design**: Choose metrics that capture what matters, not what's easy to measure
2. **Threshold selection**: Set pass thresholds based on observed performance, not guesses
3. **Boundary testing**: Test grader behavior at edge cases (empty output, None values, extreme numbers)

```python
class FinancialGrader(CodeGrader):
    def compute_metrics(self, transcript, task):
        output = transcript.final_output
        # Defensive: handle missing or malformed output
        if not output or "returns" not in output:
            return {"sharpe_ratio": 0.0, "max_drawdown": -1.0}
        ...
```

### LLMGrader Best Practices

LLMGraders are non-deterministic and prone to drift. Mitigate with:

1. **temperature=0**: Reduce variability (set in `GraderConfig`)
2. **Structured output**: Request JSON with specific fields, not free-form text
3. **Rubric anchoring**: Define what each score level means concretely
4. **Boundary examples**: Include 1-2 examples of "definitely pass" and "definitely fail" in the prompt
5. **Score normalization**: Map LLM scores (1-10) to 0-1 consistently

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

### CompositeGrader Role Selection

| Grader purpose | Role | Behavior |
|---------------|------|----------|
| Safety checks | `MUST_PASS` | Any failure = trial fails |
| Format validation | `MUST_PASS` | Any failure = trial fails |
| Quality scoring | `SCORE_CONTRIBUTOR` | Contributes to weighted average |
| Style preferences | `SCORE_CONTRIBUTOR` | Contributes to weighted average |

Use `MUST_PASS` sparingly — it creates binary signals. Use `SCORE_CONTRIBUTOR` for nuanced quality measurement.

## Human Calibration

> Note: The `human_eval/` module is planned but not yet implemented. This section describes the recommended manual process.

### Weekly 20-Sample Protocol

1. **Sample selection**: Choose 20 trials that span the score range (not just edge cases)
   - 5 from top quartile (high scores)
   - 5 from bottom quartile (low scores)
   - 10 from the middle (where grader decisions matter most)

2. **Human rating**: Rate each sample on the same rubric the LLM grader uses (1-10 per dimension)

3. **Correlation analysis**: Compute Pearson correlation between human and LLM scores
   - **> 0.8**: Excellent — grader is well-calibrated
   - **0.7-0.8**: Good — minor prompt adjustments may help
   - **< 0.7**: Action needed — review prompt, add examples, or reconsider rubric

4. **Systematic bias check**: Plot human vs LLM scores. If the LLM consistently scores higher or lower, adjust the prompt or pass threshold.

### Sampling Strategy

Avoid sampling only failures — this biases your calibration. Sample proportionally across the score distribution with slight oversampling of the boundary region (scores near the pass threshold).

## Handling Non-Determinism

### Run Count Recommendations

```python
# Minimum for any non-deterministic evaluation
config = RunnerConfig(num_runs=3)

# Recommended for production CI
config = RunnerConfig(num_runs=5)

# For benchmarking / paper results
config = RunnerConfig(num_runs=10)
```

### Report Both Statistics

Always report both pass@k (capability) and pass^k (reliability):

```python
gen = ReportGenerator(k_values=[1, 3, 5], consistency_k_values=[2, 3, 5])
report = gen.build_report(batch)
```

A high pass@k with low pass^k means the agent is capable but inconsistent — this is a different problem than an agent that simply can't solve the task.

### DecisionSpec Fingerprinting

Use `DecisionSpec` to ensure you're comparing apples to apples:

```python
from tracelens.core.decision_spec import DecisionSpec, ModelConfig, AgentSpec

spec = DecisionSpec(
    model=ModelConfig(model_id="gpt-4-turbo", temperature=0.7),
    agent=AgentSpec(agent_id="v2", version="1.2.3", git_commit="abc123"),
)

# Attach to transcript for reproducibility
transcript.decision_spec = spec
```

When comparing baselines, mismatched fingerprints warn you that configurations differ.

## Baseline Management

### Threshold Selection

Set regression thresholds based on observed variance, not arbitrary numbers:

1. Run your eval suite 5-10 times on the same agent version
2. Compute standard deviation for each metric
3. Set threshold at **2x standard deviation** — this catches real regressions while tolerating natural variance

```python
# If observed std of pass_rate is 0.03:
baseline.add_metric(
    metric_name="pass_rate",
    value=0.85,
    std=0.03,
    regression_threshold_relative=0.06,  # 2x std ≈ 6%
)
```

### Bootstrap CI for Comparisons

Use `BaselineManager.compare_to_baseline_with_ci()` for statistical rigor:

```python
comparison = manager.compare_to_baseline_with_ci(
    task_id="quality_benchmark",
    current_values={"pass_rate": [0.82, 0.85, 0.80, 0.83, 0.81]},
    confidence=0.95,
)
# Returns CI bounds, p-values, effect sizes
```

### Canary Baselines for Safety

Use `CANARY` baselines for safety-critical metrics that must never regress:

```python
manager.create_canary_baseline(
    task_id="safety_check",
    metrics={"safety_score": 0.99},
    fingerprint=spec.fingerprint,  # Tied to specific config
)
```

Canary baselines never auto-promote — they require explicit manual approval to change.

## Common Pitfalls

### 1. Grading Paths, Not Outcomes

**Wrong**: Check that the agent used specific tools in a specific order.
**Right**: Check that the final output meets requirements, regardless of how it got there.

### 2. Too Few Samples

Running 1 trial per task gives a binary signal with no statistical power. Run at least 3.

### 3. Stale Baselines

Baselines established months ago may not reflect current expectations. Set `max_age_days` in `PromotionPolicy` and review stale baselines regularly:

```python
stale = manager.list_stale_baselines()
```

### 4. LLM Grader Drift

LLM grader behavior changes when the underlying model is updated. After any model upgrade:
- Re-run calibration with human scores
- Compare old vs new model grading on the same transcripts
- Update prompts if correlation drops below 0.7

### 5. Ignoring pass^k

High pass@k can mask reliability problems. A task with pass@5 = 0.99 but pass^5 = 0.20 means the agent almost always succeeds eventually but fails 80% of the time when you need 5 consecutive successes. For production use, pass^k often matters more.

### 6. Overfitting the Eval Suite

If you only add tasks where the agent fails, the suite becomes a regression test, not a capability test. Regularly add new tasks from fresh failure cases and remove tasks that have been passing consistently for months.

## Evaluation Maturity Model

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **1 — Manual** | Ad-hoc spot checking | No automation, no baselines |
| **2 — Basic** | Automated eval suite | CodeGrader, num_runs=1, CI output |
| **3 — Statistical** | Non-determinism handled | num_runs >= 3, pass@k + pass^k, baselines |
| **4 — Calibrated** | Human-validated grading | Weekly calibration, LLMGrader correlation > 0.7 |
| **5 — Production** | Full pipeline with dashboard | HTML dashboard, regression gating, DecisionSpec tracking, canary baselines |

Most teams should aim for Level 3 initially and progress to Level 4-5 as their agent matures.
