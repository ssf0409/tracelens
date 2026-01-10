# Phase 2B: Crypto-Trading System Integration Guide

This guide covers integrating the `agent-eval` framework with the crypto-trading-system's hierarchical trading agents.

## Overview

The crypto-trading-system has a mature evaluation infrastructure with financial metrics, walk-forward validation, and regime analysis. This integration wraps existing metrics in the `agent-eval` framework to enable:
- Baseline regression detection
- CI/CD blocking on performance drops
- Unified evaluation interface
- pass@k analysis for agent decision consistency

## Directory Structure

Add this structure to your crypto-trading-system:

```
crypto-trading-system/
├── evaluation/                   # Existing evaluation code
│   ├── metrics.py                # Keep - financial calculations
│   ├── evaluator.py              # Keep - ModelEvaluator
│   ├── advanced_backtesting_suite.py  # Keep
│   ├── walk_forward.py           # Keep
│   ├── regime_analysis.py        # Keep
│   │
│   ├── framework/                # NEW - agent-eval integration
│   │   ├── __init__.py
│   │   ├── task.py               # TradingTask schema
│   │   ├── harness.py            # Evaluation orchestrator
│   │   └── non_determinism.py    # pass@k for agent decisions
│   │
│   ├── graders/                  # NEW - wrap existing metrics
│   │   ├── __init__.py
│   │   ├── financial.py          # Sharpe, Sortino, Calmar, etc.
│   │   ├── execution.py          # Latency, slippage
│   │   └── debate.py             # Debate impact grader
│   │
│   ├── baselines/                # NEW - regression baselines
│   │   └── baselines.json
│   │
│   └── ci/                       # NEW - CI integration
│       ├── runner.py
│       └── thresholds.py
│
└── pyproject.toml
```

## Step 1: Add Dependency

In `crypto-trading-system/pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies
    "agent-eval @ git+https://github.com/ssf0409/agent-eval.git",
]
```

Install:
```bash
uv sync
```

## Step 2: Define Trading Task Schema

Create `evaluation/framework/task.py`:

```python
"""Trading task schema for evaluation."""

from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field

from agent_eval.core.task import Task, TaskExpectation


class MarketConditions(BaseModel):
    """Market conditions for the trading task."""

    symbol: str = Field(description="Trading pair, e.g., BTC/USDT")
    timeframe: str = Field(description="Candle timeframe: 1m, 5m, 1h, 1d")
    start_date: datetime
    end_date: datetime
    regime: str | None = Field(default=None, description="bull, bear, sideways, volatile")


class TradingConfig(BaseModel):
    """Trading configuration."""

    initial_capital: float = 100000.0
    position_size_pct: float = 0.1
    max_positions: int = 1
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    use_debate: bool = True


class TradingTask(Task):
    """Task for evaluating trading agent performance."""

    market_conditions: MarketConditions
    trading_config: TradingConfig = Field(default_factory=TradingConfig)

    # Baseline thresholds
    min_sharpe: float = 1.0
    max_drawdown: float = -0.20  # -20%
    min_win_rate: float = 0.50

    @classmethod
    def from_json(cls, data: dict) -> "TradingTask":
        """Create task from JSON data."""
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data.get("description", ""),
            market_conditions=MarketConditions(**data["market_conditions"]),
            trading_config=TradingConfig(**data.get("trading_config", {})),
            min_sharpe=data.get("min_sharpe", 1.0),
            max_drawdown=data.get("max_drawdown", -0.20),
            min_win_rate=data.get("min_win_rate", 0.50),
            input_data={
                "market_conditions": data["market_conditions"],
                "trading_config": data.get("trading_config", {}),
            },
            expectation=TaskExpectation(
                expected_metrics={
                    "sharpe_ratio": data.get("min_sharpe", 1.0),
                    "max_drawdown": data.get("max_drawdown", -0.20),
                    "win_rate": data.get("min_win_rate", 0.50),
                },
            ),
            category=data.get("category", "backtest"),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
        )


# Predefined task templates
BTC_DAILY_BACKTEST = TradingTask(
    task_id="btc-daily-backtest",
    name="BTC Daily Backtest",
    description="Standard BTC/USDT daily backtest over 1 year",
    market_conditions=MarketConditions(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
    ),
    min_sharpe=1.2,
    max_drawdown=-0.15,
    min_win_rate=0.55,
    input_data={},
    category="backtest",
    tags=["btc", "daily", "standard"],
)

ETH_HOURLY_VOLATILE = TradingTask(
    task_id="eth-hourly-volatile",
    name="ETH Hourly Volatile Period",
    description="ETH/USDT hourly during high volatility",
    market_conditions=MarketConditions(
        symbol="ETH/USDT",
        timeframe="1h",
        start_date=datetime(2023, 3, 1),
        end_date=datetime(2023, 4, 1),
        regime="volatile",
    ),
    min_sharpe=0.8,
    max_drawdown=-0.25,
    min_win_rate=0.45,
    input_data={},
    category="backtest",
    tags=["eth", "hourly", "volatile"],
)
```

## Step 3: Create Financial Graders

Create `evaluation/graders/financial.py`:

```python
"""Financial metrics grader wrapping existing metrics.py."""

from agent_eval.core.grader import CodeGrader
from agent_eval.core.transcript import Transcript
from agent_eval.core.task import Task

# Import existing metrics
from evaluation.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
)


class FinancialMetricsGrader(CodeGrader):
    """Grade trading performance using financial metrics."""

    grader_id = "financial_metrics"
    grader_version = "1.0.0"

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task
    ) -> dict[str, float]:
        """Extract financial metrics from backtest results."""
        output = transcript.final_output

        # Expected output structure from backtest
        returns = output.get("returns", [])
        trades = output.get("trades", [])
        equity_curve = output.get("equity_curve", [])

        if not returns:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": -1.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "num_trades": 0,
            }

        return {
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "sortino_ratio": calculate_sortino_ratio(returns),
            "calmar_ratio": calculate_calmar_ratio(returns, equity_curve),
            "max_drawdown": calculate_max_drawdown(equity_curve),
            "win_rate": calculate_win_rate(trades),
            "profit_factor": calculate_profit_factor(trades),
            "total_return": (equity_curve[-1] / equity_curve[0] - 1) if equity_curve else 0,
            "num_trades": len(trades),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task
    ) -> tuple[bool, float]:
        """Determine if metrics meet thresholds."""
        # Get thresholds from task expectation
        expected = task.expectation.expected_metrics if task.expectation else {}

        min_sharpe = expected.get("sharpe_ratio", 1.0)
        max_dd = expected.get("max_drawdown", -0.20)
        min_wr = expected.get("win_rate", 0.50)

        # Check each threshold
        sharpe_ok = metrics["sharpe_ratio"] >= min_sharpe
        dd_ok = metrics["max_drawdown"] >= max_dd  # Less negative is better
        wr_ok = metrics["win_rate"] >= min_wr

        passed = sharpe_ok and dd_ok and wr_ok

        # Composite score (weighted average)
        # Normalize each metric to 0-1 scale
        sharpe_score = min(metrics["sharpe_ratio"] / 2.0, 1.0)  # 2.0 = perfect
        dd_score = 1.0 + metrics["max_drawdown"]  # -0.2 -> 0.8
        wr_score = metrics["win_rate"]

        score = (sharpe_score * 0.4 + dd_score * 0.3 + wr_score * 0.3)

        return passed, score


class ExecutionMetricsGrader(CodeGrader):
    """Grade execution quality (latency, slippage)."""

    grader_id = "execution_metrics"
    grader_version = "1.0.0"

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task
    ) -> dict[str, float]:
        """Extract execution metrics."""
        output = transcript.final_output
        trades = output.get("trades", [])

        if not trades:
            return {
                "avg_latency_ms": 0.0,
                "avg_slippage_pct": 0.0,
                "max_slippage_pct": 0.0,
                "fill_rate": 1.0,
            }

        latencies = [t.get("latency_ms", 0) for t in trades]
        slippages = [t.get("slippage_pct", 0) for t in trades]
        fills = [1 if t.get("filled", True) else 0 for t in trades]

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_slippage_pct": sum(slippages) / len(slippages),
            "max_slippage_pct": max(slippages) if slippages else 0,
            "fill_rate": sum(fills) / len(fills),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task
    ) -> tuple[bool, float]:
        """Check execution quality thresholds."""
        # Reasonable defaults
        max_latency = 100.0  # 100ms
        max_slippage = 0.01  # 1%

        latency_ok = metrics["avg_latency_ms"] <= max_latency
        slippage_ok = metrics["avg_slippage_pct"] <= max_slippage

        passed = latency_ok and slippage_ok

        # Score based on how well we beat thresholds
        latency_score = max(0, 1 - metrics["avg_latency_ms"] / max_latency)
        slippage_score = max(0, 1 - metrics["avg_slippage_pct"] / max_slippage)
        score = (latency_score + slippage_score + metrics["fill_rate"]) / 3

        return passed, score
```

## Step 4: Create Debate Impact Grader

Create `evaluation/graders/debate.py`:

```python
"""Grader for debate mechanism impact."""

from agent_eval.core.grader import CodeGrader
from agent_eval.core.transcript import Transcript
from agent_eval.core.task import Task


class DebateImpactGrader(CodeGrader):
    """Measure the impact of the debate mechanism on trading decisions."""

    grader_id = "debate_impact"
    grader_version = "1.0.0"

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task
    ) -> dict[str, float]:
        """Compute debate impact metrics."""
        output = transcript.final_output

        # Get debate-specific data
        debates = output.get("debates", [])
        decisions = output.get("decisions", [])

        if not debates:
            return {
                "debate_participation_rate": 0.0,
                "consensus_rate": 0.0,
                "decision_change_rate": 0.0,
                "debate_quality_score": 0.0,
                "avg_rounds": 0.0,
            }

        # Calculate metrics
        total_decisions = len(decisions)
        decisions_with_debate = sum(1 for d in decisions if d.get("had_debate"))
        consensus_decisions = sum(1 for d in debates if d.get("reached_consensus"))
        changed_decisions = sum(1 for d in debates if d.get("changed_initial"))

        avg_rounds = sum(d.get("rounds", 1) for d in debates) / len(debates)

        # Quality score based on argument diversity and depth
        quality_scores = [d.get("quality_score", 0.5) for d in debates]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            "debate_participation_rate": decisions_with_debate / total_decisions if total_decisions else 0,
            "consensus_rate": consensus_decisions / len(debates) if debates else 0,
            "decision_change_rate": changed_decisions / len(debates) if debates else 0,
            "debate_quality_score": avg_quality,
            "avg_rounds": avg_rounds,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task
    ) -> tuple[bool, float]:
        """Evaluate debate effectiveness."""
        # Debate should be used for important decisions
        participation_ok = metrics["debate_participation_rate"] >= 0.5

        # Should reach consensus most of the time
        consensus_ok = metrics["consensus_rate"] >= 0.7

        # Quality should be high
        quality_ok = metrics["debate_quality_score"] >= 0.6

        passed = participation_ok and consensus_ok and quality_ok

        score = (
            metrics["debate_participation_rate"] * 0.2 +
            metrics["consensus_rate"] * 0.3 +
            metrics["debate_quality_score"] * 0.5
        )

        return passed, score
```

## Step 5: Set Up Baseline Thresholds

Create `evaluation/ci/thresholds.py`:

```python
"""Regression thresholds for CI blocking."""

from agent_eval.baselines.comparison import RegressionSeverity

# Metric configurations
METRIC_THRESHOLDS = {
    "sharpe_ratio": {
        "baseline": 1.2,
        "absolute_threshold": -0.2,  # Block if drops by 0.2
        "relative_threshold": 0.10,  # Block if drops by 10%
        "higher_is_better": True,
    },
    "sortino_ratio": {
        "baseline": 1.5,
        "absolute_threshold": -0.3,
        "relative_threshold": 0.15,
        "higher_is_better": True,
    },
    "max_drawdown": {
        "baseline": -0.15,
        "absolute_threshold": -0.05,  # Block if gets 5% worse
        "relative_threshold": 0.20,
        "higher_is_better": True,  # Less negative is better
    },
    "win_rate": {
        "baseline": 0.55,
        "absolute_threshold": -0.05,
        "relative_threshold": 0.10,
        "higher_is_better": True,
    },
    "debate_improvement": {
        "baseline": 0.05,  # 5% improvement from debate
        "absolute_threshold": -0.02,
        "relative_threshold": 0.30,
        "higher_is_better": True,
    },
}

# CI blocking rules
CI_BLOCKING_RULES = {
    # Block on any MODERATE or SEVERE regression
    "default_threshold": RegressionSeverity.MODERATE,

    # Critical metrics that block on MINOR regression
    "critical_metrics": ["sharpe_ratio", "max_drawdown"],

    # Allow these to regress more before blocking
    "lenient_metrics": ["debate_improvement"],
}
```

## Step 6: Create Evaluation Harness

Create `evaluation/framework/harness.py`:

```python
"""Evaluation harness for crypto-trading agents."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any

from agent_eval.core.task import EvalSet
from agent_eval.core.trial import Trial, TrialBatch, TrialStatus
from agent_eval.core.transcript import Transcript
from agent_eval.core.grader import CompositeGrader
from agent_eval.statistics.pass_at_k import PassAtKAnalyzer
from agent_eval.statistics.consistency import ConsistencyAnalyzer
from agent_eval.baselines.manager import BaselineManager
from agent_eval.baselines.comparison import RegressionDetector

from evaluation.framework.task import TradingTask
from evaluation.graders.financial import FinancialMetricsGrader, ExecutionMetricsGrader
from evaluation.graders.debate import DebateImpactGrader
from evaluation.ci.thresholds import METRIC_THRESHOLDS, CI_BLOCKING_RULES

# Import your trading system
from agents.orchestrator import TradingOrchestrator
from backtesting.engine import BacktestEngine


class CryptoTradingEvaluator:
    """Orchestrates evaluation of trading agents."""

    def __init__(
        self,
        baselines_file: Path = Path("evaluation/baselines/baselines.json"),
        num_runs: int = 3,  # Fewer runs needed for deterministic backtests
    ):
        self.baselines_file = baselines_file
        self.num_runs = num_runs

        # Initialize components
        self.baseline_manager = BaselineManager(baselines_file)

        # Create composite grader
        self.grader = CompositeGrader(
            graders=[
                FinancialMetricsGrader(),
                ExecutionMetricsGrader(),
                DebateImpactGrader(),
            ],
            aggregation="mean",
        )

        # Statistics
        self.pass_at_k = PassAtKAnalyzer(k_values=[1, 2, 3])
        self.consistency = ConsistencyAnalyzer(k_values=[2, 3])

    def run_backtest(self, task: TradingTask) -> dict[str, Any]:
        """Run a single backtest."""
        engine = BacktestEngine(
            symbol=task.market_conditions.symbol,
            timeframe=task.market_conditions.timeframe,
            start_date=task.market_conditions.start_date,
            end_date=task.market_conditions.end_date,
            initial_capital=task.trading_config.initial_capital,
        )

        orchestrator = TradingOrchestrator(
            use_debate=task.trading_config.use_debate,
        )

        results = engine.run(orchestrator)

        return {
            "returns": results.daily_returns,
            "trades": results.trades,
            "equity_curve": results.equity_curve,
            "debates": results.debate_records,
            "decisions": results.decision_records,
        }

    async def run_single_trial(
        self,
        task: TradingTask,
        run_index: int,
    ) -> Trial:
        """Run a single evaluation trial."""
        trial = Trial(
            trial_id=f"{task.task_id}-run-{run_index}",
            task_id=task.task_id,
            run_index=run_index,
            total_runs=self.num_runs,
            status=TrialStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            transcript = Transcript(
                transcript_id=f"transcript-{trial.trial_id}",
                task_id=task.task_id,
                agent_name="trading_orchestrator",
                agent_version="1.0.0",
                started_at=datetime.utcnow(),
            )

            # Run backtest
            results = self.run_backtest(task)

            transcript.final_output = results
            transcript.completed_at = datetime.utcnow()
            trial.transcript = transcript

            # Grade results
            outcome = await self.grader.grade(transcript, task)
            trial.add_outcome(outcome)

            trial.status = TrialStatus.COMPLETED
            trial.completed_at = datetime.utcnow()

        except Exception as e:
            trial.status = TrialStatus.FAILED
            trial.error_message = str(e)
            trial.completed_at = datetime.utcnow()

        return trial

    async def run_evaluation(
        self,
        tasks: list[TradingTask],
    ) -> dict:
        """Run full evaluation suite."""
        batch = TrialBatch(
            batch_id=f"eval-{datetime.utcnow().isoformat()}",
            eval_set_id="crypto-trading",
        )

        for task in tasks:
            for run_idx in range(self.num_runs):
                trial = await self.run_single_trial(task, run_idx)
                batch.add_trial(trial)

        # Compute statistics
        pass_results = batch.get_pass_results_by_task()
        capability = self.pass_at_k.analyze(pass_results)
        reliability = self.consistency.analyze(pass_results)

        # Check regressions
        detector = RegressionDetector(min_delta_percent=5.0)
        regression_reports = {}
        should_block_ci = False

        for task_id, results in pass_results.items():
            baseline = self.baseline_manager.get_baseline(task_id)
            if baseline:
                task_trials = batch.get_trials_for_task(task_id)
                current_metrics = [
                    trial.outcomes[0].metrics
                    for trial in task_trials
                    if trial.outcomes
                ]
                report = detector.compare(baseline, current_metrics)
                regression_reports[task_id] = report

                if report.should_block_ci(CI_BLOCKING_RULES["default_threshold"]):
                    should_block_ci = True

        return {
            "batch": batch,
            "capability": capability,
            "reliability": reliability,
            "regression_reports": regression_reports,
            "should_block_ci": should_block_ci,
            "summary": self._generate_summary(batch, capability, reliability),
        }

    def update_baselines(self, batch: TrialBatch):
        """Update baselines from successful evaluation."""
        for task_id in set(t.task_id for t in batch.trials):
            trials = batch.get_trials_for_task(task_id)
            if not trials:
                continue

            # Average metrics across runs
            all_metrics = [t.outcomes[0].metrics for t in trials if t.outcomes]
            if not all_metrics:
                continue

            avg_metrics = {}
            for key in all_metrics[0]:
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = sum(values) / len(values)

            self.baseline_manager.update_baseline(
                task_id=task_id,
                metrics=avg_metrics,
            )

        self.baseline_manager.save()

    def _generate_summary(self, batch, capability, reliability) -> dict:
        trials = batch.trials
        passed = sum(1 for t in trials if t.passed)

        return {
            "total_trials": len(trials),
            "passed": passed,
            "pass_rate": passed / len(trials) if trials else 0,
            "pass@1": capability.get("pass@1", 0),
            "reliability_score": reliability.get("reliability_score", 0),
        }
```

## Step 7: Handle Non-Determinism

Create `evaluation/framework/non_determinism.py`:

```python
"""Handling non-determinism in agent decisions.

Trading agents may have non-deterministic elements:
- LLM-based analysis
- Debate outcomes
- Exploration vs exploitation

This module provides tools to analyze decision consistency.
"""

from agent_eval.statistics.pass_at_k import pass_at_k
from agent_eval.statistics.consistency import pass_to_k


def analyze_decision_consistency(
    decision_results: dict[str, list[str]],
) -> dict[str, float]:
    """Analyze how consistently the agent makes the same decisions.

    Args:
        decision_results: Dict of decision_point -> list of decisions across runs
                         e.g., {"entry_signal_1": ["buy", "buy", "hold", "buy"]}

    Returns:
        Consistency metrics
    """
    if not decision_results:
        return {"overall_consistency": 0.0}

    consistencies = []

    for point, decisions in decision_results.items():
        if not decisions:
            continue

        # Most common decision
        from collections import Counter
        counter = Counter(decisions)
        most_common_count = counter.most_common(1)[0][1]

        # Consistency = fraction of runs with same decision
        consistency = most_common_count / len(decisions)
        consistencies.append(consistency)

    return {
        "overall_consistency": sum(consistencies) / len(consistencies) if consistencies else 0,
        "min_consistency": min(consistencies) if consistencies else 0,
        "max_consistency": max(consistencies) if consistencies else 0,
        "num_decision_points": len(decision_results),
    }


def compute_outcome_stability(
    metric_results: dict[str, list[float]],
    tolerance_pct: float = 0.05,
) -> dict[str, float]:
    """Measure how stable outcomes are across runs.

    Args:
        metric_results: Dict of metric_name -> list of values across runs
        tolerance_pct: Consider values "same" if within this percentage

    Returns:
        Stability metrics
    """
    import numpy as np

    stabilities = {}

    for metric, values in metric_results.items():
        if not values:
            stabilities[metric] = 0.0
            continue

        mean = np.mean(values)
        std = np.std(values)

        # Coefficient of variation (lower = more stable)
        cv = std / abs(mean) if mean != 0 else float("inf")

        # What fraction of values are within tolerance of mean
        within_tolerance = sum(
            1 for v in values
            if abs(v - mean) / abs(mean) <= tolerance_pct
        ) / len(values) if mean != 0 else 0

        stabilities[metric] = {
            "cv": cv,
            "std": std,
            "within_tolerance": within_tolerance,
            "is_stable": cv < tolerance_pct,
        }

    return stabilities
```

## Step 8: Run Evaluation

```python
import asyncio
from evaluation.framework.harness import CryptoTradingEvaluator
from evaluation.framework.task import BTC_DAILY_BACKTEST, ETH_HOURLY_VOLATILE

async def main():
    evaluator = CryptoTradingEvaluator(num_runs=3)

    tasks = [BTC_DAILY_BACKTEST, ETH_HOURLY_VOLATILE]

    results = await evaluator.run_evaluation(tasks)

    print(f"Pass Rate: {results['summary']['pass_rate']:.2%}")
    print(f"pass@1: {results['summary']['pass@1']:.2%}")

    # Check CI status
    if results["should_block_ci"]:
        print("CI BLOCKED: Regression detected!")
        for task_id, report in results["regression_reports"].items():
            if report.has_regression:
                print(f"  {task_id}: {report.overall_severity}")
                print(report.to_ci_output())
        exit(1)

    # Update baselines on success
    evaluator.update_baselines(results["batch"])
    print("Baselines updated successfully")

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria

1. **Sharpe ratio** ≥ 1.2 (no regression > 10%)
2. **Max drawdown** ≤ -15% (no worsening > 5%)
3. **Win rate** ≥ 55% (no regression > 10%)
4. **Debate improvement** ≥ 5% (debate should help)
5. **Outcome stability** CV < 5% across runs
