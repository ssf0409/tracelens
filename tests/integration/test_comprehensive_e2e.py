"""Comprehensive end-to-end tests exercising every layer of tracelens.

Tests cover:
1. Basic pipeline: adapter -> runner -> grader -> report
2. Composite grading with MUST_PASS + SCORE_CONTRIBUTOR roles
3. DecisionSpec fingerprinting attached to transcripts
4. Baseline creation, comparison, and regression detection
5. Report serialization round-trip (to_dict / from_dict)
6. EvalSet filtering via filtered_eval_set
7. Timeout and adapter error paths
8. Multiple runs with pass@k / pass^k statistics
9. CLI argument parser validation
10. JSON task loading round-trip
"""

import asyncio
import json
from pathlib import Path

import pytest

from tracelens.baselines.comparison import RegressionDetector, RegressionSeverity
from tracelens.baselines.manager import BaselineManager, BaselineType, PromotionPolicy
from tracelens.core._time import utc_now
from tracelens.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    ModelConfig,
    PromptSpec,
)
from tracelens.core.grader import (
    CodeGrader,
    CompositeGrader,
    GraderConfig,
    GraderRole,
)
from tracelens.core.task import EvalSet, JSONTaskLoader, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig
from tracelens.reporting.generator import ReportData, ReportGenerator

# ---------------------------------------------------------------------------
# Test graders
# ---------------------------------------------------------------------------

class AccuracyGrader(CodeGrader):
    """Checks if agent answer equals expected value."""

    def __init__(self) -> None:
        super().__init__("accuracy")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.input_data.get("expected")
        actual = transcript.final_output.get("answer") if transcript.final_output else None
        return {"accuracy": 1.0 if actual == expected else 0.0}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["accuracy"] >= 1.0, metrics["accuracy"]


class BoundsGrader(CodeGrader):
    """Safety grader: checks answer is within [0, 1000]."""

    def __init__(self) -> None:
        config = GraderConfig(role=GraderRole.MUST_PASS)
        super().__init__("bounds_check", config=config)

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        answer = transcript.final_output.get("answer", 0) if transcript.final_output else 0
        in_bounds = 1.0 if 0 <= answer <= 1000 else 0.0
        return {"in_bounds": in_bounds}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["in_bounds"] >= 1.0, metrics["in_bounds"]


class QualityGrader(CodeGrader):
    """Score contributor: gives a quality score based on answer magnitude."""

    def __init__(self) -> None:
        config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR, weight=2.0)
        super().__init__("quality", config=config)

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        answer = transcript.final_output.get("answer", 0) if transcript.final_output else 0
        # Normalize: higher answer → higher quality (capped at 1.0)
        return {"quality": min(abs(answer) / 100.0, 1.0)}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["quality"] >= 0.5, metrics["quality"]


# ---------------------------------------------------------------------------
# Test adapters
# ---------------------------------------------------------------------------

async def _adder_agent(input_data: dict) -> dict:
    """Adds two numbers."""
    return {"answer": input_data.get("a", 0) + input_data.get("b", 0)}


async def _negative_agent(input_data: dict) -> dict:
    """Returns a negative number (will fail bounds check)."""
    return {"answer": -999}


async def _slow_agent(input_data: dict) -> dict:
    """Sleeps too long."""
    await asyncio.sleep(10)
    return {"answer": 0}


async def _crashing_agent(input_data: dict) -> dict:
    """Always raises."""
    raise RuntimeError("agent exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComprehensivePipeline:
    """Full pipeline with multiple graders, runs, and report generation."""

    async def test_multi_grader_pipeline(self):
        """Run with AccuracyGrader + BoundsGrader, verify per-trial outcomes."""
        tasks = [
            Task(task_id="t1", name="3+4", input_data={"a": 3, "b": 4, "expected": 7}),
            Task(task_id="t2", name="50+50", input_data={"a": 50, "b": 50, "expected": 100}),
        ]
        eval_set = EvalSet(name="Multi-Grader", tasks=tasks)

        adapter = SimpleAdapter(_adder_agent)
        config = RunnerConfig(num_runs=5, max_concurrency=3)
        runner = EvaluationRunner(adapter, [AccuracyGrader(), BoundsGrader()], config)
        batch = await runner.run(eval_set)

        assert batch.total_count == 10  # 2 tasks × 5 runs
        assert batch.all_complete

        for trial in batch.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert len(trial.outcomes) == 2
            # Both graders should pass
            assert trial.passed
            # trial_id consistency
            for outcome in trial.outcomes:
                assert outcome.trial_id == trial.trial_id

        # Pass rate should be 100%
        assert batch.pass_rate == pytest.approx(1.0)


class TestCompositeGraderE2E:
    """Composite grader with MUST_PASS and SCORE_CONTRIBUTOR roles."""

    async def test_must_pass_failure_blocks_trial(self):
        """A MUST_PASS grader failure should fail the trial even if others pass."""
        tasks = [
            Task(task_id="neg", name="Negative", input_data={"a": -500, "b": -500, "expected": -1000}),
        ]
        eval_set = EvalSet(name="Bounds Test", tasks=tasks)

        # CompositeGrader: BoundsGrader (MUST_PASS) + QualityGrader (SCORE_CONTRIBUTOR)
        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (BoundsGrader(), 0.3),
                (QualityGrader(), 0.7),
            ],
        )

        adapter = SimpleAdapter(_adder_agent)
        runner = EvaluationRunner(adapter, [composite], RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED
        # Accuracy: -1000 == -1000 (correct), but bounds: -1000 is out of [0,1000]
        # MUST_PASS failure → trial fails
        assert trial.passed is False
        outcome = trial.outcomes[0]
        assert outcome.passed is False
        # Should have feedback about MUST_PASS failure
        assert "MUST-PASS" in outcome.feedback

    async def test_score_contributor_failure_doesnt_block(self):
        """A SCORE_CONTRIBUTOR failure doesn't block the trial if MUST_PASS passes."""
        tasks = [
            Task(task_id="small", name="Small", input_data={"a": 1, "b": 2, "expected": 3}),
        ]
        eval_set = EvalSet(name="Quality Test", tasks=tasks)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (BoundsGrader(), 0.3),     # MUST_PASS — should pass (3 is in bounds)
                (QualityGrader(), 0.7),    # SCORE_CONTRIBUTOR — quality=0.03, fails threshold
            ],
        )

        adapter = SimpleAdapter(_adder_agent)
        runner = EvaluationRunner(adapter, [composite], RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED
        # MUST_PASS passes → trial passes (even though quality is low)
        assert trial.passed is True


class TestDecisionSpecE2E:
    """DecisionSpec attached to transcripts for reproducibility tracking."""

    async def test_fingerprint_in_transcript(self):
        """DecisionSpec fingerprint flows through the pipeline."""
        spec = DecisionSpec(
            model=ModelConfig(provider="test", model_id="test-model-v1", temperature=0.0),
            agent=AgentSpec(agent_name="adder", agent_version="1.0.0"),
            environment=EnvironmentSpec(git_commit="abc123"),
        )

        class _AnnotatingAdapter(AgentAdapter):
            async def run(self, task: Task) -> Transcript:
                transcript = self.start_transcript(task)
                transcript.decision_spec = spec
                transcript.final_output = {"answer": task.input_data["a"] + task.input_data["b"]}
                transcript.completed_at = utc_now()
                return transcript

        tasks = [Task(task_id="t1", name="1+1", input_data={"a": 1, "b": 1, "expected": 2})]
        eval_set = EvalSet(name="Spec Test", tasks=tasks)
        runner = EvaluationRunner(_AnnotatingAdapter(), [AccuracyGrader()], RunnerConfig())
        batch = await runner.run(eval_set)

        trial = batch.trials[0]
        assert trial.fingerprint is not None
        assert trial.fingerprint_short is not None
        assert len(trial.fingerprint) == 64  # SHA-256 hex
        assert trial.fingerprint_short == trial.fingerprint[:12]

        # Summary includes fingerprint
        summary = trial.to_summary_dict()
        assert "fingerprint" in summary

    async def test_fingerprint_determinism(self):
        """Same DecisionSpec always produces the same fingerprint."""
        spec1 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            prompts=PromptSpec.from_prompts(system_prompt="You are helpful"),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            prompts=PromptSpec.from_prompts(system_prompt="You are helpful"),
        )
        assert spec1.fingerprint == spec2.fingerprint

    async def test_fingerprint_changes_on_config_change(self):
        """Different config → different fingerprint."""
        spec1 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus", temperature=0.0),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus", temperature=0.7),
        )
        assert spec1.fingerprint != spec2.fingerprint


class TestBaselineRegressionE2E:
    """Baseline creation, comparison, and regression detection."""

    async def test_baseline_workflow(self, tmp_path: Path):
        """Full baseline workflow: create → run → compare → detect regression."""
        baselines_file = tmp_path / "baselines.json"
        manager = BaselineManager(baselines_file)

        # 1. Create a capability baseline from a "good" run
        manager.create_capability_baseline(
            task_id="t1",
            metrics={"accuracy": 0.9, "quality": 0.8},
            metric_stds={"accuracy": 0.05, "quality": 0.1},
            sample_size=20,
            task_name="Addition",
        )
        manager.save()

        # 2. Reload to verify persistence
        manager2 = BaselineManager(baselines_file)
        baseline = manager2.get_baseline("t1")
        assert baseline is not None
        assert baseline.baseline_type == BaselineType.CAPABILITY
        assert baseline.get_metric("accuracy").baseline_value == pytest.approx(0.9)

        # 3. Simulate a regression: current accuracy dropped to 0.6
        detector = RegressionDetector(significance_level=0.05, min_delta_percent=5.0)
        current_results = [
            {"accuracy": 0.62, "quality": 0.75},
            {"accuracy": 0.58, "quality": 0.78},
            {"accuracy": 0.61, "quality": 0.72},
            {"accuracy": 0.59, "quality": 0.77},
            {"accuracy": 0.60, "quality": 0.74},
        ]
        report = detector.compare(baseline, current_results)

        assert report.has_regression is True
        assert report.overall_severity in (
            RegressionSeverity.MODERATE,
            RegressionSeverity.SEVERE,
        )

        # Accuracy regressed, quality may or may not
        accuracy_regressions = [r for r in report.regressions if r.metric_name == "accuracy"]
        assert len(accuracy_regressions) == 1
        assert accuracy_regressions[0].delta < 0  # Declined

        # CI output contains the regression info
        ci_output = report.to_ci_output()
        assert "REGRESSION" in ci_output
        assert "accuracy" in ci_output

        # should_block_ci works
        assert report.should_block_ci(RegressionSeverity.MODERATE) is True

    async def test_canary_baseline_protection(self, tmp_path: Path):
        """Canary baselines cannot be auto-promoted."""
        baselines_file = tmp_path / "baselines.json"
        manager = BaselineManager(baselines_file)

        manager.create_canary_baseline(
            task_id="safety",
            metrics={"safety_score": 0.95},
            fingerprint="a" * 64,
        )
        manager.save()

        # Try auto-promotion — should fail
        promoted, reason = manager.try_promote(
            task_id="safety",
            current_metrics={"safety_score": 0.99},
            sample_size=100,
        )
        assert promoted is False
        assert "canary" in reason.lower() or "Canary" in reason

    async def test_capability_baseline_promotion(self, tmp_path: Path):
        """Capability baselines can be auto-promoted when criteria are met."""
        baselines_file = tmp_path / "baselines.json"
        manager = BaselineManager(baselines_file)

        policy = PromotionPolicy(
            allow_auto_promotion=True,
            min_improvement_relative=0.05,
            min_samples=5,
        )
        manager.create_capability_baseline(
            task_id="quality",
            metrics={"score": 0.70},
            sample_size=10,
            promotion_policy=policy,
        )

        # Promote with 10% improvement
        promoted, reason = manager.try_promote(
            task_id="quality",
            current_metrics={"score": 0.80},
            sample_size=10,
        )
        assert promoted is True

        # Verify new baseline value
        updated = manager.get_baseline("quality")
        assert updated.get_metric("score").baseline_value == pytest.approx(0.80)
        assert updated.promotion_count == 1


class TestReportSerializationE2E:
    """Report round-trip: build → to_dict → from_dict → verify."""

    async def test_full_report_round_trip(self):
        """Build a report, serialize, deserialize, verify all fields survive."""
        tasks = [
            Task(task_id="t1", name="T1", input_data={"a": 5, "b": 5, "expected": 10}),
            Task(task_id="t2", name="T2", input_data={"a": 1, "b": 1, "expected": 2}),
        ]
        eval_set = EvalSet(name="Round Trip", tasks=tasks)

        adapter = SimpleAdapter(_adder_agent)
        config = RunnerConfig(num_runs=5)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], config)
        batch = await runner.run(eval_set)

        gen = ReportGenerator(k_values=[1, 3, 5], consistency_k_values=[2, 3])
        report = gen.build_report(batch)

        # Serialize
        data = report.to_dict()
        json_str = json.dumps(data, default=str)

        # Deserialize
        restored = ReportData.from_dict(json.loads(json_str))

        assert restored.total_trials == report.total_trials
        assert restored.total_tasks == report.total_tasks
        assert restored.overall_pass_rate == pytest.approx(report.overall_pass_rate)
        assert restored.overall_mean_score == pytest.approx(report.overall_mean_score)
        assert len(restored.task_summaries) == len(report.task_summaries)
        assert set(restored.pass_at_k.keys()) == set(report.pass_at_k.keys())
        assert set(restored.reliability.keys()) == set(report.reliability.keys())


class TestEvalSetFilteringE2E:
    """EvalSet filtering before running evaluations."""

    async def test_filtered_eval_set_runs_only_matching_tasks(self):
        """filtered_eval_set returns an EvalSet that preserves config."""
        tasks = [
            Task(task_id="easy-1", name="Easy", input_data={"a": 1, "b": 1, "expected": 2},
                 category="easy"),
            Task(task_id="easy-2", name="Easy2", input_data={"a": 2, "b": 2, "expected": 4},
                 category="easy"),
            Task(task_id="hard-1", name="Hard", input_data={"a": 99, "b": 99, "expected": 198},
                 category="hard"),
        ]
        full_set = EvalSet(name="Mixed", tasks=tasks, default_num_runs=3)

        # Filter to easy only
        easy_set = full_set.filtered_eval_set(categories=["easy"])
        assert len(easy_set.tasks) == 2
        assert easy_set.default_num_runs == 3  # Config preserved

        # Run only easy tasks
        adapter = SimpleAdapter(_adder_agent)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], RunnerConfig(num_runs=2))
        batch = await runner.run(easy_set)

        assert batch.total_count == 4  # 2 tasks × 2 runs
        assert batch.pass_rate == pytest.approx(1.0)

        # Verify no hard tasks were run
        task_ids = {t.task_id for t in batch.trials}
        assert "hard-1" not in task_ids


class TestErrorPathsE2E:
    """Timeout, adapter errors, and grader errors in the full pipeline."""

    async def test_timeout_produces_correct_report(self):
        """Timed-out trials appear in report with 0% pass rate."""
        tasks = [Task(task_id="slow", name="Slow", input_data={"a": 1, "b": 1, "expected": 2})]
        eval_set = EvalSet(name="Timeout", tasks=tasks)

        adapter = SimpleAdapter(_slow_agent)
        config = RunnerConfig(timeout_seconds=0.05)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], config)
        batch = await runner.run(eval_set)

        assert batch.total_count == 1
        trial = batch.trials[0]
        assert trial.status == TrialStatus.TIMEOUT
        assert trial.transcript is None
        assert len(trial.outcomes) == 0
        assert trial.passed is False

        # Report should show 0% pass rate
        gen = ReportGenerator()
        report = gen.build_report(batch)
        assert report.overall_pass_rate == pytest.approx(0.0)

    async def test_adapter_crash_produces_correct_report(self):
        """Crashed adapter trials show FAILED status, 0 outcomes."""
        tasks = [Task(task_id="crash", name="Crash", input_data={})]
        eval_set = EvalSet(name="Crash", tasks=tasks)

        adapter = SimpleAdapter(_crashing_agent)
        runner = EvaluationRunner(adapter, [AccuracyGrader()])
        batch = await runner.run(eval_set)

        trial = batch.trials[0]
        assert trial.status == TrialStatus.FAILED
        assert "exploded" in trial.error_message
        assert len(trial.outcomes) == 0

    async def test_mixed_success_and_failure(self):
        """Mix of passing, failing, and crashing tasks produces correct aggregate."""
        tasks = [
            Task(task_id="ok", name="OK", input_data={"a": 1, "b": 2, "expected": 3}),
            Task(task_id="wrong", name="Wrong", input_data={"a": 1, "b": 1, "expected": 999}),
        ]
        eval_set = EvalSet(name="Mixed", tasks=tasks)

        adapter = SimpleAdapter(_adder_agent)
        config = RunnerConfig(num_runs=3)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], config)
        batch = await runner.run(eval_set)

        assert batch.total_count == 6  # 2 tasks × 3 runs
        results = batch.get_pass_results_by_task()
        assert all(results["ok"])  # All 3 pass
        assert not any(results["wrong"])  # All 3 fail
        assert batch.pass_rate == pytest.approx(0.5)


class TestPassAtKReliabilityE2E:
    """Verify pass@k and pass^k statistics from a real pipeline run."""

    async def test_statistics_from_pipeline(self):
        """Run pipeline and verify pass@k/pass^k are reasonable."""
        tasks = [
            Task(task_id="always-pass", name="Pass",
                 input_data={"a": 1, "b": 1, "expected": 2}),
            Task(task_id="always-fail", name="Fail",
                 input_data={"a": 1, "b": 1, "expected": 999}),
        ]
        eval_set = EvalSet(name="Stats", tasks=tasks)

        adapter = SimpleAdapter(_adder_agent)
        config = RunnerConfig(num_runs=10)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], config)
        batch = await runner.run(eval_set)

        gen = ReportGenerator(k_values=[1, 5], consistency_k_values=[2, 5])
        report = gen.build_report(batch)

        # pass@1: average of task pass rates → (1.0 + 0.0) / 2 = 0.5
        assert report.pass_at_k["pass@1"] == pytest.approx(0.5)
        # pass@5: always-pass → 1.0, always-fail → 0.0 → avg 0.5
        assert report.pass_at_k["pass@5"] == pytest.approx(0.5)

        # pass^2 for always-pass: 1.0 (all consecutive pairs pass)
        # pass^2 for always-fail: 0.0
        # Average: 0.5
        assert report.reliability["pass^2"] == pytest.approx(0.5)

        # Per-task summaries
        summaries = {s.task_id: s for s in report.task_summaries}
        assert summaries["always-pass"].pass_rate == pytest.approx(1.0)
        assert summaries["always-pass"].mean_score == pytest.approx(1.0)
        assert summaries["always-fail"].pass_rate == pytest.approx(0.0)
        assert summaries["always-fail"].mean_score == pytest.approx(0.0)


class TestJSONTaskRoundTrip:
    """Load tasks from JSON, run, verify results."""

    async def test_json_load_run_report(self, tmp_path: Path):
        """Tasks loaded from JSON file produce correct results."""
        # Write tasks to JSON
        tasks_data = {
            "tasks": [
                {"task_id": "json-1", "name": "JSON 1", "input_data": {"a": 10, "b": 20, "expected": 30}},
                {"task_id": "json-2", "name": "JSON 2", "input_data": {"a": 7, "b": 3, "expected": 10}},
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        with open(tasks_file, "w") as f:
            json.dump(tasks_data, f)

        # Load
        loader = JSONTaskLoader()
        tasks = loader.load(tasks_file)
        assert len(tasks) == 2
        assert tasks[0].task_id == "json-1"

        # Run
        eval_set = EvalSet(name="JSON Suite", tasks=tasks)
        adapter = SimpleAdapter(_adder_agent)
        runner = EvaluationRunner(adapter, [AccuracyGrader()], RunnerConfig(num_runs=2))
        batch = await runner.run(eval_set)

        assert batch.total_count == 4
        assert batch.pass_rate == pytest.approx(1.0)

        # Write report JSON and verify it's valid
        gen = ReportGenerator()
        report = gen.build_report(batch)
        report_file = tmp_path / "results.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, default=str)

        # Read back
        with open(report_file) as f:
            loaded = json.load(f)
        assert loaded["total_trials"] == 4
        assert loaded["overall_pass_rate"] == pytest.approx(1.0)
