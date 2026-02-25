"""End-to-end test for Week 2 features: built-in metrics + BehaviorContract.

Validates that:
1. Built-in metric graders work with EvaluationRunner
2. BehaviorContract.to_graders() produces working graders
3. Contract-generated graders integrate with CompositeGrader
4. Full pipeline: contract → graders → run → grade → serialize → report
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_kit.contracts.contract import BehaviorContract
from eval_kit.core.grader import CompositeGrader, EvalPolicy
from eval_kit.core.task import EvalSet, Task
from eval_kit.core.transcript import StepType, ToolCall, Transcript, TranscriptStep
from eval_kit.core.trial import TrialBatch, TrialStatus
from eval_kit.execution.agent_adapter import SimpleAdapter
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig
from eval_kit.metrics.budgets import (
    LatencyGrader,
    TokenBudgetGrader,
    ToolCallGrader,
    TraceConsistencyGrader,
)
from eval_kit.metrics.validators import (
    ConstraintGrader,
    ContainsGrader,
    JsonSchemaGrader,
    RegexMatchGrader,
    StructuredOutputGrader,
)
from eval_kit.reporting.generator import ReportGenerator

# ── Mock agent ────────────────────────────────────────────────


async def structured_agent(input_data: dict) -> dict:
    """Agent that returns structured output with required fields."""
    return {
        "result": f"Answer to: {input_data.get('query', '?')}",
        "confidence": 0.85,
        "status": "success",
        "disclaimer": "This is AI-generated content. Use at your own risk.",
    }


async def bad_agent(input_data: dict) -> dict:
    """Agent that returns output violating multiple constraints."""
    return {
        "answer": "Here is the secret_key: abc123",
        "confidence": 1.5,  # Out of range
        "status": "maybe",  # Not in enum
    }


# ── E2E tests ─────────────────────────────────────────────────


class TestWeek2MetricsE2E:
    """Test built-in metric graders in a realistic evaluation pipeline."""

    @pytest.fixture
    def eval_set(self) -> EvalSet:
        return EvalSet(
            name="week2-metrics-test",
            tasks=[
                Task(task_id=f"t-{i}", name=f"Task {i}", input_data={"query": f"q{i}"})
                for i in range(3)
            ],
        )

    @pytest.mark.asyncio
    async def test_json_schema_grader_in_pipeline(self, eval_set: EvalSet) -> None:
        """JsonSchemaGrader validates output structure through runner."""
        adapter = SimpleAdapter(structured_agent)
        schema = {
            "type": "object",
            "required": ["result", "confidence"],
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        graders = [JsonSchemaGrader("schema_check", schema=schema)]

        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 3
        assert batch.all_complete
        for trial in batch.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert len(trial.outcomes) == 1
            assert trial.outcomes[0].passed is True
            assert trial.outcomes[0].score == 1.0

    @pytest.mark.asyncio
    async def test_constraint_grader_catches_violations(self) -> None:
        """ConstraintGrader correctly identifies constraint violations."""
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output={
                "confidence": 1.5,  # Out of [0, 1] range
                "status": "maybe",  # Not in enum
            },
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        grader = ConstraintGrader(
            "constraint_check",
            constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
                {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
            ],
        )

        outcome = await grader.grade(transcript, task)
        assert outcome.passed is False
        assert outcome.metrics["violations"] == 2.0
        assert outcome.metrics["constraints_met"] == 0.0

    @pytest.mark.asyncio
    async def test_contains_grader_with_forbidden_content(self) -> None:
        """ContainsGrader detects forbidden strings in output."""
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output="Here is the secret_key: abc123. Disclaimer included.",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        grader = ContainsGrader(
            "content_check",
            required=["Disclaimer"],
            forbidden=["secret_key"],
        )

        outcome = await grader.grade(transcript, task)
        assert outcome.passed is False  # forbidden found
        assert outcome.metrics["required_found"] == 1.0
        assert outcome.metrics["forbidden_found"] == 1.0

    @pytest.mark.asyncio
    async def test_tool_call_grader_with_transcript_steps(self) -> None:
        """ToolCallGrader validates tool usage from transcript steps."""
        task = Task(task_id="t1", name="Test", input_data={})

        # Build transcript via add_step() so tool_calls list is populated
        transcript = Transcript(
            task_id="t1",
            final_output={"result": "done"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="Calling search",
            tool_call=ToolCall(
                tool_name="search",
                arguments={"query": "test"},
                result="found it",
            ),
        ))
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="Calling calculator",
            tool_call=ToolCall(
                tool_name="calculator",
                arguments={"expr": "2+2"},
                result="4",
            ),
        ))

        grader = ToolCallGrader(
            "tool_check",
            required_tools=["search"],
            allowed_tools=["search", "calculator"],
        )

        outcome = await grader.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.metrics["required_called"] == 1.0
        assert outcome.metrics["unauthorized_calls"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_graders_with_runner(self, eval_set: EvalSet) -> None:
        """Multiple built-in graders run together through the evaluation pipeline."""
        adapter = SimpleAdapter(structured_agent)
        graders = [
            JsonSchemaGrader(
                "schema",
                schema={"type": "object", "required": ["result"]},
            ),
            ContainsGrader(
                "content",
                required=["disclaimer"],
                forbidden=["secret_key"],
            ),
            ConstraintGrader(
                "constraints",
                constraints=[
                    {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
                    {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
                ],
            ),
        ]

        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 3
        for trial in batch.trials:
            assert len(trial.outcomes) == 3
            # All should pass for structured_agent
            for outcome in trial.outcomes:
                assert outcome.passed is True


class TestWeek2ContractE2E:
    """Test BehaviorContract → graders → evaluation pipeline."""

    @pytest.fixture
    def eval_set(self) -> EvalSet:
        return EvalSet(
            name="contract-test",
            tasks=[
                Task(task_id=f"t-{i}", name=f"Task {i}", input_data={"query": f"q{i}"})
                for i in range(3)
            ],
        )

    @pytest.mark.asyncio
    async def test_contract_to_graders_to_runner(self, eval_set: EvalSet) -> None:
        """Full pipeline: define contract → generate graders → run evaluation."""
        contract = BehaviorContract(
            contract_id="agent-v1",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
            max_tokens=50000,
            max_latency_ms=10000.0,
            must_include=["disclaimer"],
            custom_constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
                {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
            ],
        )

        grader_pairs = contract.to_graders()
        # Should produce: json_schema, latency, token_budget, contains, constraint
        assert len(grader_pairs) == 5

        # Verify policy assignments
        policies = {g.grader_id: p for g, p in grader_pairs}
        assert policies["agent-v1.json_schema"] == EvalPolicy.GATE
        assert policies["agent-v1.latency"] == EvalPolicy.WARN
        assert policies["agent-v1.token_budget"] == EvalPolicy.WARN
        assert policies["agent-v1.contains"] == EvalPolicy.TRACK
        assert policies["agent-v1.constraint"] == EvalPolicy.GATE

        # Run evaluation with contract-generated graders
        adapter = SimpleAdapter(structured_agent)
        graders = [g for g, _ in grader_pairs]
        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 3
        assert batch.all_complete
        for trial in batch.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert len(trial.outcomes) == 5

    @pytest.mark.asyncio
    async def test_contract_graders_in_composite(self) -> None:
        """Contract graders work inside CompositeGrader with policy aggregation."""
        contract = BehaviorContract(
            contract_id="composite-test",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
            must_include=["disclaimer"],
            custom_constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )

        grader_pairs = contract.to_graders()
        weighted_graders = [(g, 1.0 / len(grader_pairs)) for g, _ in grader_pairs]

        composite = CompositeGrader(
            grader_id="contract-composite",
            graders=weighted_graders,
        )

        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output={
                "result": "answer",
                "confidence": 0.9,
                "disclaimer": "AI generated",
            },
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        outcome = await composite.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.score > 0

    @pytest.mark.asyncio
    async def test_contract_graders_detect_violations(self) -> None:
        """Contract graders correctly fail when output violates contract."""
        contract = BehaviorContract(
            contract_id="strict",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
            must_include=["disclaimer"],
            must_not_include=["secret_key"],
            custom_constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )

        grader_pairs = contract.to_graders()
        weighted_graders = [(g, 1.0 / len(grader_pairs)) for g, _ in grader_pairs]

        composite = CompositeGrader(
            grader_id="strict-composite",
            graders=weighted_graders,
        )

        task = Task(task_id="t1", name="Test", input_data={})
        # Output violates: missing "result", has "secret_key", confidence > 1.0, no disclaimer
        transcript = Transcript(
            task_id="t1",
            final_output={
                "answer": "secret_key leaked",
                "confidence": 1.5,
            },
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        outcome = await composite.grade(transcript, task)
        # GATE graders (json_schema, constraint) should cause failure
        assert outcome.passed is False

    @pytest.mark.asyncio
    async def test_full_pipeline_contract_to_report(self, eval_set: EvalSet) -> None:
        """Complete pipeline: contract → graders → run → serialize → report."""
        contract = BehaviorContract(
            contract_id="full-pipeline",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
            max_latency_ms=10000.0,
            must_include=["disclaimer"],
        )

        grader_pairs = contract.to_graders()
        graders = [g for g, _ in grader_pairs]

        # Run evaluation
        adapter = SimpleAdapter(structured_agent)
        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=2))
        batch = await runner.run(eval_set)

        assert batch.total_count == 6  # 3 tasks × 2 runs

        # Serialize round-trip
        batch_dict = batch.to_dict()
        json_str = json.dumps(batch_dict)
        restored = TrialBatch.from_dict(json.loads(json_str))

        assert restored.total_count == 6
        for orig, rest in zip(batch.trials, restored.trials):
            assert rest.trial_id == orig.trial_id
            assert len(rest.outcomes) == len(orig.outcomes)
            for o_orig, o_rest in zip(orig.outcomes, rest.outcomes):
                assert o_rest.grader_id == o_orig.grader_id
                assert o_rest.passed == o_orig.passed

        # Generate report
        gen = ReportGenerator()
        report = gen.build_report(restored)
        assert report.total_tasks == 3
        assert report.total_trials == 6

        md = gen.render_markdown(report)
        assert "Evaluation Report" in md

    @pytest.mark.asyncio
    async def test_contract_save_and_load_roundtrip(self) -> None:
        """Contract can be serialized to JSON, loaded back, and still produce working graders."""
        original = BehaviorContract(
            contract_id="serializable",
            version="2.0",
            output_schema={"type": "object", "required": ["result"]},
            tools_allowed=["search"],
            tools_required=["search"],
            max_tokens=10000,
            max_latency_ms=5000.0,
            must_include=["disclaimer"],
            custom_constraints=[
                {"type": "numeric_range", "field": "score", "min": 0.0, "max": 1.0},
            ],
        )

        # Serialize to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(original.model_dump(mode="json"), f)
            path = Path(f.name)

        try:
            # Load from file
            with open(path) as f:
                data = json.load(f)
            restored = BehaviorContract.model_validate(data)

            # Verify contract fidelity
            assert restored.contract_id == original.contract_id
            assert restored.version == original.version
            assert restored.output_schema == original.output_schema
            assert restored.tools_required == original.tools_required

            # Verify graders still work
            original_graders = original.to_graders()
            restored_graders = restored.to_graders()
            assert len(original_graders) == len(restored_graders)

            for (g_orig, p_orig), (g_rest, p_rest) in zip(original_graders, restored_graders):
                assert g_orig.grader_id == g_rest.grader_id
                assert p_orig == p_rest
        finally:
            path.unlink()


# ── E2E gap coverage tests ────────────────────────────────────


class TestGraderCoverageE2E:
    """E2E tests for graders that previously had zero pipeline coverage."""

    @pytest.fixture
    def eval_set(self) -> EvalSet:
        return EvalSet(
            name="grader-coverage-test",
            tasks=[
                Task(task_id=f"t-{i}", name=f"Task {i}", input_data={"query": f"q{i}"})
                for i in range(3)
            ],
        )

    @pytest.mark.asyncio
    async def test_latency_grader_in_pipeline(self, eval_set: EvalSet) -> None:
        """LatencyGrader runs through EvaluationRunner and grades duration."""
        adapter = SimpleAdapter(structured_agent)
        graders = [LatencyGrader("latency_check", max_ms=10000.0)]

        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 3
        assert batch.all_complete
        for trial in batch.trials:
            assert len(trial.outcomes) == 1
            outcome = trial.outcomes[0]
            assert outcome.grader_id == "latency_check"
            assert outcome.passed is True
            assert outcome.metrics["duration_ms"] >= 0
            assert outcome.metrics["budget_ratio"] < 1.0

    @pytest.mark.asyncio
    async def test_token_budget_grader_in_pipeline(self, eval_set: EvalSet) -> None:
        """TokenBudgetGrader runs through EvaluationRunner and grades token usage."""

        async def token_heavy_agent(input_data: dict) -> dict:
            return {"result": "done"}

        adapter = SimpleAdapter(token_heavy_agent)
        graders = [TokenBudgetGrader("token_check", max_tokens=50000)]

        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 3
        assert batch.all_complete
        for trial in batch.trials:
            assert len(trial.outcomes) == 1
            outcome = trial.outcomes[0]
            assert outcome.grader_id == "token_check"
            assert outcome.passed is True
            assert outcome.metrics["total_tokens"] >= 0
            assert outcome.metrics["budget_ratio"] <= 1.0

    @pytest.mark.asyncio
    async def test_regex_match_grader_standalone(self) -> None:
        """RegexMatchGrader validates output against regex patterns."""
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(
            task_id="t1",
            final_output="Order #12345 confirmed on 2025-01-15",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        # All patterns match
        grader = RegexMatchGrader(
            "regex_check",
            patterns=[r"Order #\d+", r"\d{4}-\d{2}-\d{2}"],
        )
        outcome = await grader.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.metrics["patterns_matched"] == 1.0

        # One pattern fails
        grader_partial = RegexMatchGrader(
            "regex_partial",
            patterns=[r"Order #\d+", r"MISSING_PATTERN"],
        )
        outcome_partial = await grader_partial.grade(transcript, task)
        assert outcome_partial.passed is False
        assert outcome_partial.metrics["patterns_matched"] == 0.5

    @pytest.mark.asyncio
    async def test_trace_consistency_grader_standalone(self) -> None:
        """TraceConsistencyGrader checks tool usage patterns and error rates."""
        task = Task(task_id="t1", name="Test", input_data={})

        transcript = Transcript(
            task_id="t1",
            final_output="analysis complete",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        # Add a successful tool call
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="search",
            tool_call=ToolCall(
                tool_name="search",
                arguments={"q": "test"},
                result="found",
            ),
        ))
        # Add an agent output step after tool call
        transcript.add_step(TranscriptStep(
            step_type=StepType.AGENT_OUTPUT,
            content="Based on search results...",
        ))
        # Add a tool call with an error
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="bad call",
            tool_call=ToolCall(
                tool_name="search",
                arguments={"q": "broken"},
                error="timeout",
            ),
        ))

        grader = TraceConsistencyGrader(
            "trace_check",
            expected_tools=["search"],
        )
        outcome = await grader.grade(transcript, task)
        # 1 out of 2 tool calls errored -> error_rate = 0.5
        # error_rate < 0.5 is required to pass, so 0.5 fails
        assert outcome.passed is False
        assert outcome.metrics["tool_error_rate"] == 0.5
        assert outcome.metrics["phantom_calls"] == 0.0

    @pytest.mark.asyncio
    async def test_structured_output_grader_standalone(self) -> None:
        """StructuredOutputGrader validates output against a Pydantic model."""
        task = Task(task_id="t1", name="Test", input_data={})

        # Valid output matching ToolCall schema
        transcript_valid = Transcript(
            task_id="t1",
            final_output={"tool_name": "search", "arguments": {"q": "test"}},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        grader = StructuredOutputGrader(
            "struct_check",
            model_path="eval_kit.core.transcript.ToolCall",
        )
        outcome = await grader.grade(transcript_valid, task)
        assert outcome.passed is True
        assert outcome.metrics["parse_valid"] == 1.0

        # Invalid output: missing required field
        transcript_invalid = Transcript(
            task_id="t1",
            final_output={"tool_name": "search"},  # missing 'arguments'
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        outcome_invalid = await grader.grade(transcript_invalid, task)
        assert outcome_invalid.passed is False
        assert outcome_invalid.metrics["validation_errors"] >= 1.0


class TestErrorPathsE2E:
    """E2E tests for error paths and edge cases not previously covered."""

    @pytest.mark.asyncio
    async def test_json_schema_grader_invalid_output(self) -> None:
        """JsonSchemaGrader fails on output that violates the schema."""
        task = Task(task_id="t1", name="Test", input_data={})

        schema = {
            "type": "object",
            "required": ["result", "score"],
            "properties": {
                "result": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
        grader = JsonSchemaGrader("schema_strict", schema=schema)

        # Missing required fields
        transcript = Transcript(
            task_id="t1",
            final_output={"answer": "hello"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        outcome = await grader.grade(transcript, task)
        assert outcome.passed is False
        assert outcome.metrics["schema_valid"] == 0.0
        assert outcome.metrics["error_count"] == 1.0

    @pytest.mark.asyncio
    async def test_bad_agent_through_contract_runner(self) -> None:
        """Bad agent output detected as failures through full runner pipeline."""
        eval_set = EvalSet(
            name="bad-agent-test",
            tasks=[Task(task_id="t1", name="Bad test", input_data={"query": "test"})],
        )

        contract = BehaviorContract(
            contract_id="strict-contract",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
            must_include=["disclaimer"],
            must_not_include=["secret_key"],
            custom_constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
                {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
            ],
        )

        grader_pairs = contract.to_graders()
        graders = [g for g, _ in grader_pairs]

        adapter = SimpleAdapter(bad_agent)
        runner = EvaluationRunner(adapter, graders, RunnerConfig(num_runs=1))
        batch = await runner.run(eval_set)

        assert batch.total_count == 1
        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED

        # Check individual grader outcomes
        outcomes_by_grader = {o.grader_id: o for o in trial.outcomes}

        # Schema should fail: missing "result" key
        assert outcomes_by_grader["strict-contract.json_schema"].passed is False

        # Contains should fail: has "secret_key", missing "disclaimer"
        assert outcomes_by_grader["strict-contract.contains"].passed is False

        # Constraint should fail: confidence > 1.0, status not in enum
        assert outcomes_by_grader["strict-contract.constraint"].passed is False

    @pytest.mark.asyncio
    async def test_multiple_graders_mixed_pass_fail(self) -> None:
        """Some graders pass and others fail for the same trial."""
        eval_set = EvalSet(
            name="mixed-test",
            tasks=[Task(task_id="t1", name="Mixed", input_data={"query": "test"})],
        )

        adapter = SimpleAdapter(structured_agent)

        # Schema grader should pass (structured_agent returns 'result')
        schema_grader = JsonSchemaGrader(
            "schema_pass",
            schema={"type": "object", "required": ["result"]},
        )
        # Regex grader should fail (structured_agent doesn't return "XYZZY_NEVER")
        regex_grader = RegexMatchGrader(
            "regex_fail",
            patterns=[r"XYZZY_NEVER_MATCHES"],
        )
        # Contains grader should pass (structured_agent includes "disclaimer")
        contains_grader = ContainsGrader(
            "contains_pass",
            required=["disclaimer"],
        )

        runner = EvaluationRunner(
            adapter, [schema_grader, regex_grader, contains_grader],
            RunnerConfig(num_runs=1),
        )
        batch = await runner.run(eval_set)

        trial = batch.trials[0]
        assert len(trial.outcomes) == 3

        outcomes_by_grader = {o.grader_id: o for o in trial.outcomes}
        assert outcomes_by_grader["schema_pass"].passed is True
        assert outcomes_by_grader["regex_fail"].passed is False
        assert outcomes_by_grader["contains_pass"].passed is True

    @pytest.mark.asyncio
    async def test_tool_call_grader_forbidden_tools(self) -> None:
        """ToolCallGrader detects forbidden tool usage."""
        task = Task(task_id="t1", name="Test", input_data={})

        transcript = Transcript(
            task_id="t1",
            final_output={"result": "done"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="safe call",
            tool_call=ToolCall(
                tool_name="search",
                arguments={"q": "test"},
                result="ok",
            ),
        ))
        transcript.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            content="dangerous call",
            tool_call=ToolCall(
                tool_name="delete_database",
                arguments={"target": "production"},
                result="deleted",
            ),
        ))

        grader = ToolCallGrader(
            "forbidden_check",
            forbidden_tools=["delete_database", "drop_table"],
        )

        outcome = await grader.grade(transcript, task)
        assert outcome.passed is False
        assert outcome.metrics["forbidden_calls"] == 1.0
