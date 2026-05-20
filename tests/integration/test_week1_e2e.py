"""End-to-end test for Week 1 features: persistence, eval policy, LLM provider.

Validates that all three Week 1 features work together in a realistic workflow:
1. Run evaluation with policy-aware graders (GATE/WARN/TRACK)
2. LLM grader uses InMemoryProvider
3. Serialize TrialBatch to JSON and back
4. Report groups results by policy
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tracelens.core.grader import (
    CodeGrader,
    CompositeGrader,
    EvalPolicy,
    GraderConfig,
    GraderRole,
    LLMGrader,
)
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig
from tracelens.llm.provider import InMemoryProvider
from tracelens.reporting.generator import ReportGenerator

# ── Test graders ───────────────────────────────────────────────


class SafetyGrader(CodeGrader):
    """Gate grader: checks output is a dict with 'result' key."""

    def __init__(self) -> None:
        super().__init__("safety", config=GraderConfig(policy=EvalPolicy.GATE))

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        output = transcript.final_output
        has_result = 1.0 if isinstance(output, dict) and "result" in output else 0.0
        return {"has_result_key": has_result}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        passed = metrics["has_result_key"] == 1.0
        return passed, metrics["has_result_key"]


class LatencyCheckGrader(CodeGrader):
    """Warn grader: flags slow transcripts."""

    def __init__(self) -> None:
        super().__init__(
            "latency_check",
            config=GraderConfig(policy=EvalPolicy.WARN, threshold=1000.0),
        )

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        duration = transcript.duration_ms or 0.0
        return {"duration_ms": duration}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        threshold = self.config.threshold or 1000.0
        passed = metrics["duration_ms"] <= threshold
        score = max(0.0, min(1.0, 1.0 - metrics["duration_ms"] / (threshold * 2)))
        return passed, score


class QualityLLMGrader(LLMGrader):
    """Track grader: LLM-based quality using InMemoryProvider."""

    def __init__(self, provider: InMemoryProvider) -> None:
        super().__init__(
            "quality_llm",
            provider=provider,
            config=GraderConfig(policy=EvalPolicy.TRACK),
        )

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        return f"Rate quality of: {transcript.final_output}"

    def parse_llm_response(
        self, response: str, task: Task
    ) -> tuple[bool, float, dict[str, float], str]:
        data = json.loads(response)
        score = data["score"] / 10.0
        return score >= 0.7, score, {"raw_score": data["score"]}, data["feedback"]


# ── Test adapter ───────────────────────────────────────────────


async def mock_agent(input_data: dict) -> dict:
    """Simple mock agent that returns structured output."""
    return {"result": f"Processed: {input_data.get('query', 'unknown')}"}


# ── E2E test ───────────────────────────────────────────────────


class TestWeek1E2E:
    """End-to-end test combining persistence, eval policy, and LLM provider."""

    @pytest.fixture
    def eval_set(self) -> EvalSet:
        tasks = [
            Task(task_id=f"task-{i}", name=f"Task {i}", input_data={"query": f"q{i}"})
            for i in range(3)
        ]
        return EvalSet(name="week1-test", tasks=tasks)

    @pytest.fixture
    def llm_provider(self) -> InMemoryProvider:
        return InMemoryProvider(responses=[
            '{"score": 8, "feedback": "Good quality"}',
            '{"score": 6, "feedback": "Acceptable"}',
            '{"score": 9, "feedback": "Excellent"}',
        ])

    @pytest.mark.asyncio
    async def test_full_eval_with_policy_graders_and_persistence(
        self, eval_set: EvalSet, llm_provider: InMemoryProvider
    ) -> None:
        """Run eval → grade with GATE/WARN/TRACK → serialize → deserialize → report."""

        # 1. Set up adapter and graders
        adapter = SimpleAdapter(mock_agent)
        graders = [
            SafetyGrader(),          # GATE
            LatencyCheckGrader(),    # WARN
            QualityLLMGrader(llm_provider),  # TRACK (uses InMemoryProvider)
        ]

        # Verify policies are set correctly
        assert graders[0].policy == EvalPolicy.GATE
        assert graders[1].policy == EvalPolicy.WARN
        assert graders[2].policy == EvalPolicy.TRACK

        # 2. Run evaluation
        config = RunnerConfig(num_runs=1, max_concurrency=3, timeout_seconds=10)
        runner = EvaluationRunner(adapter, graders, config)
        batch = await runner.run(eval_set)

        # 3. Verify basic results
        assert batch.total_count == 3
        assert batch.all_complete
        for trial in batch.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert len(trial.outcomes) == 3  # One per grader
            assert trial.transcript is not None

        # 4. Verify LLM provider was called
        assert len(llm_provider.prompts) == 3
        assert all("Rate quality of:" in p for p in llm_provider.prompts)

        # 5. Serialize to dict → JSON string → back
        batch_dict = batch.to_dict()
        json_str = json.dumps(batch_dict)
        parsed = json.loads(json_str)
        restored = TrialBatch.from_dict(parsed)

        # 6. Verify round-trip fidelity
        assert restored.batch_id == batch.batch_id
        assert len(restored.trials) == 3
        for orig, rest in zip(batch.trials, restored.trials):
            assert rest.trial_id == orig.trial_id
            assert rest.task_id == orig.task_id
            assert rest.status == orig.status
            assert len(rest.outcomes) == len(orig.outcomes)
            for o_orig, o_rest in zip(orig.outcomes, rest.outcomes):
                assert o_rest.grader_id == o_orig.grader_id
                assert o_rest.passed == o_orig.passed
                assert o_rest.score == pytest.approx(o_orig.score, abs=1e-6)

            # Transcript survives
            assert rest.transcript is not None
            assert rest.transcript.final_output == orig.transcript.final_output

        # 7. Generate report from restored batch
        gen = ReportGenerator()
        report = gen.build_report(restored)
        assert report.total_tasks == 3
        assert report.total_trials == 3

        # Markdown renders without error
        md = gen.render_markdown(report)
        assert "Evaluation Report" in md

    @pytest.mark.asyncio
    async def test_save_trials_to_file_and_reload(
        self, eval_set: EvalSet, llm_provider: InMemoryProvider
    ) -> None:
        """Simulate --save-trials: write to file, reload, generate report."""
        adapter = SimpleAdapter(mock_agent)
        graders = [SafetyGrader(), QualityLLMGrader(llm_provider)]

        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=2))
        batch = await runner.run(eval_set)

        # Write to temp file (simulates --save-trials)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch.to_dict(), f, indent=2)
            path = Path(f.name)

        try:
            # Reload from file
            with open(path) as f:
                data = json.load(f)
            restored = TrialBatch.from_dict(data)

            assert restored.total_count == 6  # 3 tasks × 2 runs
            assert restored.pass_rate == batch.pass_rate

            # Generate report from file-loaded data
            gen = ReportGenerator()
            report = gen.build_report(restored)
            assert report.total_tasks == 3
            assert report.total_trials == 6
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_composite_grader_policy_integration(self) -> None:
        """CompositeGrader with mixed policies: GATE failure blocks, WARN/TRACK don't."""
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output={"wrong_key": "no result key"},  # Will fail safety gate
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        provider = InMemoryProvider(responses=['{"score": 9, "feedback": "Great"}'])

        composite = CompositeGrader(
            grader_id="combined",
            graders=[
                (SafetyGrader(), 0.4),           # GATE — will FAIL
                (LatencyCheckGrader(), 0.3),      # WARN — will pass (0ms)
                (QualityLLMGrader(provider), 0.3),  # TRACK — will pass
            ],
        )

        outcome = await composite.grade(transcript, task)

        # Overall should FAIL because GATE grader failed
        assert outcome.passed is False
        # Score still computed (weighted average)
        assert outcome.score > 0
        # Failure tracked in metrics
        assert outcome.metrics.get("_failed_must_pass", 0) >= 1

    @pytest.mark.asyncio
    async def test_backward_compat_must_pass_with_new_policy(self) -> None:
        """Legacy MUST_PASS role still blocks alongside new GATE policy."""
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output={"result": "valid"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        # Legacy must-pass grader (should still block)
        class LegacyMustPassGrader(CodeGrader):
            def __init__(self) -> None:
                super().__init__(
                    "legacy_safety",
                    config=GraderConfig(role=GraderRole.MUST_PASS),
                )

            def compute_metrics(self, transcript, task):
                return {"check": 0.0}

            def determine_pass(self, metrics, task):
                return False, 0.0  # Always fails

        composite = CompositeGrader(
            grader_id="mixed",
            graders=[
                (LegacyMustPassGrader(), 0.5),  # Legacy MUST_PASS — fails
                (SafetyGrader(), 0.5),           # New GATE — passes
            ],
        )

        outcome = await composite.grade(transcript, task)
        # Legacy MUST_PASS failure still blocks
        assert outcome.passed is False
