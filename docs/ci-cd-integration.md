# Phase 3: CI/CD Integration Guide

This guide covers setting up automated evaluation in CI/CD pipelines with regression blocking and baseline management.

## Overview

The CI/CD integration provides:
- **Automated evaluation** on every PR
- **Regression detection** with severity-based blocking
- **Baseline updates** on main branch merges
- **Human calibration** workflow (weekly)

## GitHub Actions Workflow

### StrideAI Workflow

Create `.github/workflows/eval.yml` in your StrideAI project:

```yaml
name: Agent Evaluation

on:
  pull_request:
    branches: [main]
    paths:
      - 'app/agents/**'
      - 'eval/**'
      - 'pyproject.toml'
  push:
    branches: [main]
  schedule:
    # Weekly full evaluation (Sunday at midnight)
    - cron: '0 0 * * 0'
  workflow_dispatch:
    inputs:
      full_eval:
        description: 'Run full evaluation (all scenarios, more runs)'
        type: boolean
        default: false
      update_baselines:
        description: 'Update baselines after evaluation'
        type: boolean
        default: false

env:
  PYTHON_VERSION: '3.12'
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

jobs:
  quick-eval:
    name: Quick Evaluation
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run quick evaluation
        run: |
          uv run python -m eval.harness \
            --mode quick \
            --num-runs 3 \
            --categories programming,career \
            --baseline-check \
            --fail-on-regression moderate

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-quick
          path: eval/results/

      - name: Comment on PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval/results/summary.json'));

            const body = `## Evaluation Results

            | Metric | Value |
            |--------|-------|
            | Pass Rate | ${(results.pass_rate * 100).toFixed(1)}% |
            | pass@1 | ${(results.pass_at_1 * 100).toFixed(1)}% |
            | pass@5 | ${(results.pass_at_5 * 100).toFixed(1)}% |
            | Reliability | ${(results.reliability * 100).toFixed(1)}% |

            ${results.has_regression ? '⚠️ **Regression Detected**' : '✅ No Regressions'}

            <details>
            <summary>Details</summary>

            ${results.details}
            </details>
            `;

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });

  full-eval:
    name: Full Evaluation
    runs-on: ubuntu-latest
    if: |
      github.event_name == 'push' ||
      github.event_name == 'schedule' ||
      github.event.inputs.full_eval == 'true'

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run full evaluation
        run: |
          uv run python -m eval.harness \
            --mode full \
            --num-runs 10 \
            --baseline-check \
            --fail-on-regression moderate

      - name: Update baselines
        if: |
          github.event_name == 'push' ||
          github.event.inputs.update_baselines == 'true'
        run: |
          uv run python -m eval.harness --update-baselines

          # Commit baseline updates
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add eval/baselines/
          git diff --staged --quiet || git commit -m "chore: update evaluation baselines"
          git push

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-full
          path: eval/results/

  human-calibration:
    name: Human Calibration Sample
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Generate calibration samples
        run: |
          uv run python -m eval.human_eval.sampler \
            --size 20 \
            --strategy diverse \
            --output eval/human_eval/samples/

      - name: Create calibration issue
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const samples = JSON.parse(fs.readFileSync('eval/human_eval/samples/manifest.json'));

            const body = `## Weekly Human Calibration

            20 samples are ready for human evaluation.

            ### Instructions
            1. Go to the [Calibration UI](https://your-streamlit-app.url)
            2. Review each sample and rate on 5 dimensions
            3. Submit ratings by Friday

            ### Samples
            ${samples.map(s => `- [ ] ${s.task_id}: ${s.description}`).join('\n')}

            ### Previous Calibration
            - LLM-Human Correlation: ${samples.previous_correlation || 'N/A'}
            `;

            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `[Calibration] Week of ${new Date().toISOString().split('T')[0]}`,
              body: body,
              labels: ['calibration', 'human-eval']
            });
```

### Crypto-Trading Workflow

Create `.github/workflows/eval.yml` in your crypto-trading-system:

```yaml
name: Trading Evaluation

on:
  pull_request:
    branches: [main]
    paths:
      - 'agents/**'
      - 'evaluation/**'
      - 'backtesting/**'
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      update_baselines:
        description: 'Update baselines after evaluation'
        type: boolean
        default: false

env:
  PYTHON_VERSION: '3.12'

jobs:
  backtest-eval:
    name: Backtest Evaluation
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync

      - name: Run evaluation
        run: |
          uv run python -m evaluation.ci.runner \
            --tasks btc-daily,eth-hourly \
            --num-runs 3 \
            --baseline-check \
            --fail-on-regression moderate

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: evaluation/results/

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('evaluation/results/summary.json'));

            const body = `## Trading Evaluation Results

            | Metric | Value | Threshold | Status |
            |--------|-------|-----------|--------|
            | Sharpe Ratio | ${results.sharpe.toFixed(2)} | ≥ 1.2 | ${results.sharpe >= 1.2 ? '✅' : '❌'} |
            | Max Drawdown | ${(results.max_drawdown * 100).toFixed(1)}% | ≥ -15% | ${results.max_drawdown >= -0.15 ? '✅' : '❌'} |
            | Win Rate | ${(results.win_rate * 100).toFixed(1)}% | ≥ 55% | ${results.win_rate >= 0.55 ? '✅' : '❌'} |

            ${results.has_regression ? '⚠️ **Regression Detected**: ' + results.regression_details : '✅ No Regressions'}
            `;

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });

      - name: Update baselines
        if: |
          github.event_name == 'push' ||
          github.event.inputs.update_baselines == 'true'
        run: |
          uv run python -m evaluation.ci.runner --update-baselines

          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add evaluation/baselines/
          git diff --staged --quiet || git commit -m "chore: update trading baselines"
          git push
```

## CI Runner Script

Create a CI runner script that can be invoked from the workflow.

### StrideAI CI Runner

Create `eval/__main__.py`:

```python
"""CLI entry point for evaluation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from eval.harness import StrideAIEvaluator
from agent_eval.baselines.comparison import RegressionSeverity


def parse_args():
    parser = argparse.ArgumentParser(description="Run StrideAI evaluation")

    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per task",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to evaluate",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags to filter",
    )
    parser.add_argument(
        "--baseline-check",
        action="store_true",
        help="Check for regressions against baselines",
    )
    parser.add_argument(
        "--fail-on-regression",
        type=str,
        choices=["minor", "moderate", "severe"],
        default="moderate",
        help="Minimum severity to fail CI",
    )
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update baselines from current results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Output directory for results",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse categories and tags
    categories = args.categories.split(",") if args.categories else None
    tags = args.tags.split(",") if args.tags else None

    # Initialize evaluator
    evaluator = StrideAIEvaluator(num_runs=args.num_runs)

    # Run evaluation
    print(f"Running {args.mode} evaluation with {args.num_runs} runs...")
    results = await evaluator.run_evaluation(
        categories=categories,
        tags=tags,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pass_rate": results["summary"]["pass_rate"],
        "pass_at_1": results["capability"].get("pass@1", 0),
        "pass_at_5": results["capability"].get("pass@5", 0),
        "reliability": results["reliability"].get("reliability_score", 0),
        "has_regression": False,
        "details": "",
    }

    # Check regressions
    severity_map = {
        "minor": RegressionSeverity.MINOR,
        "moderate": RegressionSeverity.MODERATE,
        "severe": RegressionSeverity.SEVERE,
    }
    threshold = severity_map[args.fail_on_regression]

    should_fail = False
    regression_details = []

    if args.baseline_check:
        for task_id, report in results["regression_reports"].items():
            if report.has_regression:
                summary["has_regression"] = True
                regression_details.append(f"{task_id}: {report.overall_severity.value}")

                if report.should_block_ci(threshold):
                    should_fail = True

    summary["details"] = "\n".join(regression_details)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\nPass Rate: {summary['pass_rate']:.2%}")
    print(f"pass@1: {summary['pass_at_1']:.2%}")
    print(f"pass@5: {summary['pass_at_5']:.2%}")

    if summary["has_regression"]:
        print(f"\nRegressions detected:")
        for detail in regression_details:
            print(f"  - {detail}")

    # Update baselines if requested
    if args.update_baselines:
        print("\nUpdating baselines...")
        evaluator.update_baselines(results["batch"])
        print("Baselines updated.")

    # Exit with error if regression blocks CI
    if should_fail:
        print(f"\nCI BLOCKED: Regression severity >= {args.fail_on_regression}")
        sys.exit(1)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Crypto-Trading CI Runner

Create `evaluation/ci/runner.py`:

```python
"""CI runner for trading evaluation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evaluation.framework.harness import CryptoTradingEvaluator
from evaluation.framework.task import BTC_DAILY_BACKTEST, ETH_HOURLY_VOLATILE
from agent_eval.baselines.comparison import RegressionSeverity


TASK_MAP = {
    "btc-daily": BTC_DAILY_BACKTEST,
    "eth-hourly": ETH_HOURLY_VOLATILE,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run trading evaluation")

    parser.add_argument(
        "--tasks",
        type=str,
        default="btc-daily",
        help="Comma-separated task IDs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per task",
    )
    parser.add_argument(
        "--baseline-check",
        action="store_true",
        help="Check for regressions",
    )
    parser.add_argument(
        "--fail-on-regression",
        type=str,
        choices=["minor", "moderate", "severe"],
        default="moderate",
        help="Minimum severity to fail CI",
    )
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update baselines from results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Output directory",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse tasks
    task_ids = args.tasks.split(",")
    tasks = [TASK_MAP[tid.strip()] for tid in task_ids if tid.strip() in TASK_MAP]

    if not tasks:
        print(f"No valid tasks found. Available: {list(TASK_MAP.keys())}")
        sys.exit(1)

    evaluator = CryptoTradingEvaluator(num_runs=args.num_runs)

    print(f"Running evaluation on {len(tasks)} tasks with {args.num_runs} runs...")
    results = await evaluator.run_evaluation(tasks)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics
    batch = results["batch"]
    all_metrics = []

    for trial in batch.trials:
        if trial.outcomes:
            all_metrics.append(trial.outcomes[0].metrics)

    if all_metrics:
        avg_sharpe = sum(m.get("sharpe_ratio", 0) for m in all_metrics) / len(all_metrics)
        avg_dd = sum(m.get("max_drawdown", 0) for m in all_metrics) / len(all_metrics)
        avg_wr = sum(m.get("win_rate", 0) for m in all_metrics) / len(all_metrics)
    else:
        avg_sharpe = avg_dd = avg_wr = 0

    summary = {
        "sharpe": avg_sharpe,
        "max_drawdown": avg_dd,
        "win_rate": avg_wr,
        "has_regression": results["should_block_ci"],
        "regression_details": "",
    }

    # Check specific regressions
    severity_map = {
        "minor": RegressionSeverity.MINOR,
        "moderate": RegressionSeverity.MODERATE,
        "severe": RegressionSeverity.SEVERE,
    }
    threshold = severity_map[args.fail_on_regression]

    regression_details = []
    should_fail = False

    if args.baseline_check:
        for task_id, report in results["regression_reports"].items():
            if report.has_regression:
                regression_details.append(f"{task_id}: {report.overall_severity.value}")
                if report.should_block_ci(threshold):
                    should_fail = True

    summary["regression_details"] = ", ".join(regression_details)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\nSharpe Ratio: {avg_sharpe:.2f}")
    print(f"Max Drawdown: {avg_dd:.2%}")
    print(f"Win Rate: {avg_wr:.2%}")

    if regression_details:
        print(f"\nRegressions: {', '.join(regression_details)}")

    if args.update_baselines:
        evaluator.update_baselines(batch)
        print("\nBaselines updated.")

    if should_fail:
        print(f"\nCI BLOCKED: Regression >= {args.fail_on_regression}")
        sys.exit(1)

    print("\nEvaluation passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Baseline Management Strategy

### When to Update Baselines

1. **On main branch push**: Automatically after successful evaluation
2. **Weekly (scheduled)**: Full evaluation with baseline update
3. **Manual trigger**: Via `workflow_dispatch` when needed

### Baseline Versioning

Baselines are versioned and tracked in git:

```json
{
  "task_id": "btc-daily-backtest",
  "version": "1.2.0",
  "updated_at": "2024-01-15T10:30:00Z",
  "git_commit": "abc123",
  "metrics": {
    "sharpe_ratio": {
      "baseline_value": 1.35,
      "std_deviation": 0.15,
      "sample_size": 30
    }
  },
  "previous_versions": [
    {"version": "1.1.0", "sharpe_ratio": 1.28}
  ]
}
```

### Handling Intentional Regressions

Sometimes you intentionally accept a regression (e.g., trading speed for safety):

```yaml
# In PR description or commit message
# EVAL:ACCEPT-REGRESSION:sharpe_ratio:reason=Added safety checks that reduce aggressiveness
```

The CI workflow can parse this and skip blocking:

```python
def should_block(report, pr_description):
    if f"EVAL:ACCEPT-REGRESSION:{report.metric_name}" in pr_description:
        return False
    return report.should_block_ci()
```

## Human Calibration Workflow

### Weekly Process

1. **Saturday (automated)**: Generate 20 diverse samples
2. **Monday-Friday**: Human reviewers rate samples in Streamlit UI
3. **Friday (automated)**: Reconcile LLM vs human scores
4. **If correlation < 0.7**: Create issue to update grader prompts

### Streamlit Calibration UI

Create `eval/human_eval/app.py`:

```python
"""Streamlit app for human calibration."""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Agent Calibration", layout="wide")

# Load samples
samples_dir = Path("eval/human_eval/samples")
samples = json.loads((samples_dir / "manifest.json").read_text())

st.title("Weekly Calibration")
st.write(f"Rate {len(samples)} samples on 5 dimensions")

# Sample selector
sample_idx = st.selectbox(
    "Select sample",
    range(len(samples)),
    format_func=lambda i: f"{i+1}. {samples[i]['task_id']}"
)

sample = samples[sample_idx]

# Display sample
col1, col2 = st.columns(2)

with col1:
    st.subheader("Goal")
    st.write(sample["goal"])

    st.subheader("User Context")
    st.json(sample["user_context"])

with col2:
    st.subheader("Agent Output")
    st.json(sample["agent_output"])

# Rating sliders
st.subheader("Your Ratings")

ratings = {}
for dimension in ["specificity", "personalization", "actionability", "constraint_respect", "overall"]:
    ratings[dimension] = st.slider(
        dimension.replace("_", " ").title(),
        1, 10, 5,
        key=f"rating_{sample_idx}_{dimension}"
    )

feedback = st.text_area("Qualitative Feedback (optional)")

if st.button("Submit Rating"):
    # Save rating
    rating_file = samples_dir / f"ratings/{sample['task_id']}.json"
    rating_file.parent.mkdir(exist_ok=True)

    rating_data = {
        "task_id": sample["task_id"],
        "ratings": ratings,
        "feedback": feedback,
        "reviewer": st.session_state.get("reviewer", "anonymous"),
    }

    rating_file.write_text(json.dumps(rating_data, indent=2))
    st.success("Rating saved!")

# Show LLM score after human rates (to avoid bias)
if st.checkbox("Show LLM Score (after rating)"):
    st.subheader("LLM Scores")
    st.json(sample.get("llm_scores", {}))
```

## Best Practices

1. **Start small**: Begin with 5 scenarios, expand to 20+
2. **Version everything**: Baselines, grader prompts, scenarios
3. **Read transcripts**: Manually review failures to catch false signals
4. **Calibrate regularly**: Weekly human eval prevents grader drift
5. **Document regressions**: Track why regressions were accepted
6. **Separate quick/full**: Quick for PRs, full for nightly/weekly
