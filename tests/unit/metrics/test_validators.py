"""Tests for validator metric graders.

Covers JsonSchemaGrader, StructuredOutputGrader, ContainsGrader,
RegexMatchGrader, and ConstraintGrader.
"""

from __future__ import annotations

import asyncio

import pytest

from tracelens.core.grader import EvalPolicy, GraderConfig
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.metrics.validators import (
    ConstraintGrader,
    ContainsGrader,
    JsonSchemaGrader,
    RegexMatchGrader,
    StructuredOutputGrader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(final_output: object) -> Transcript:
    """Create a minimal transcript with the given final_output."""
    return Transcript(task_id="task-1", final_output=final_output)


def _make_task() -> Task:
    """Create a minimal task for grading."""
    return Task(name="test-task", input_data={"prompt": "do something"})


def _run(coro):  # noqa: ANN001, ANN202
    """Convenience wrapper to run an async coroutine synchronously.

    Uses ``asyncio.run`` rather than ``get_event_loop().run_until_complete``
    so it does not depend on a current event loop being set — pytest-asyncio
    unsets the loop after async tests, which makes the legacy pattern raise.
    """
    return asyncio.run(coro)


# ===========================================================================
# JsonSchemaGrader
# ===========================================================================

class TestJsonSchemaGrader:
    """Tests for JsonSchemaGrader."""

    SIMPLE_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    def test_default_policy_is_gate(self) -> None:
        grader = JsonSchemaGrader("json-schema", schema=self.SIMPLE_SCHEMA)
        assert grader.policy == EvalPolicy.GATE

    def test_custom_policy_override(self) -> None:
        config = GraderConfig(policy=EvalPolicy.WARN)
        grader = JsonSchemaGrader(
            "json-schema", schema=self.SIMPLE_SCHEMA, config=config,
        )
        assert grader.policy == EvalPolicy.WARN

    def test_valid_output_passes(self) -> None:
        grader = JsonSchemaGrader("json-schema", schema=self.SIMPLE_SCHEMA)
        transcript = _make_transcript({"name": "Alice", "age": 30})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0
        assert outcome.metrics["schema_valid"] == 1.0
        assert outcome.metrics["error_count"] == 0.0

    def test_invalid_output_fails(self) -> None:
        grader = JsonSchemaGrader("json-schema", schema=self.SIMPLE_SCHEMA)
        # Missing required "age" field
        transcript = _make_transcript({"name": "Alice"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.score == 0.0
        assert outcome.metrics["schema_valid"] == 0.0
        assert outcome.metrics["error_count"] >= 1.0

    def test_wrong_type_fails(self) -> None:
        grader = JsonSchemaGrader("json-schema", schema=self.SIMPLE_SCHEMA)
        transcript = _make_transcript({"name": "Alice", "age": "thirty"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["schema_valid"] == 0.0

    def test_none_output_fails(self) -> None:
        grader = JsonSchemaGrader("json-schema", schema=self.SIMPLE_SCHEMA)
        transcript = _make_transcript(None)
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["schema_valid"] == 0.0

    def test_array_schema(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        grader = JsonSchemaGrader("json-schema", schema=schema)

        valid = _make_transcript([1, 2, 3])
        invalid = _make_transcript([1, "two", 3])

        assert _run(grader.grade(valid, _make_task())).passed is True
        assert _run(grader.grade(invalid, _make_task())).passed is False

    def test_invalid_schema_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="invalid schema"):
            JsonSchemaGrader(
                "bad-schema",
                schema={"type": "not_a_real_type"},
            )


# ===========================================================================
# StructuredOutputGrader
# ===========================================================================

class TestStructuredOutputGrader:
    """Tests for StructuredOutputGrader."""

    # We use a well-known Pydantic model from pydantic itself for testing
    MODEL_PATH = "tracelens.core.task.TaskExpectation"

    def test_default_policy_is_gate(self) -> None:
        grader = StructuredOutputGrader("structured", model_path=self.MODEL_PATH)
        assert grader.policy == EvalPolicy.GATE

    def test_valid_pydantic_data_passes(self) -> None:
        grader = StructuredOutputGrader("structured", model_path=self.MODEL_PATH)
        transcript = _make_transcript({"expected_output": "hello"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0
        assert outcome.metrics["parse_valid"] == 1.0
        assert outcome.metrics["validation_errors"] == 0.0

    def test_invalid_data_fails(self) -> None:
        # TaskExpectation has expected_metrics: dict[str, float] | None
        # Passing an int for that field should cause a validation error
        grader = StructuredOutputGrader("structured", model_path=self.MODEL_PATH)
        transcript = _make_transcript({"expected_metrics": "not-a-dict"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["parse_valid"] == 0.0
        assert outcome.metrics["validation_errors"] >= 1.0

    def test_non_dict_output_fails(self) -> None:
        grader = StructuredOutputGrader("structured", model_path=self.MODEL_PATH)
        transcript = _make_transcript("just a string")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["parse_valid"] == 0.0

    def test_bad_model_path_raises_runtime_error(self) -> None:
        grader = StructuredOutputGrader(
            "structured", model_path="nonexistent.module.Model",
        )
        transcript = _make_transcript({"foo": "bar"})
        with pytest.raises(RuntimeError, match="cannot load model"):
            _run(grader.grade(transcript, _make_task()))


# ===========================================================================
# ContainsGrader
# ===========================================================================

class TestContainsGrader:
    """Tests for ContainsGrader."""

    def test_default_policy_is_track(self) -> None:
        grader = ContainsGrader("contains", required=["hello"])
        assert grader.policy == EvalPolicy.TRACK

    def test_all_required_present(self) -> None:
        grader = ContainsGrader("contains", required=["hello", "world"])
        transcript = _make_transcript("hello world")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0
        assert outcome.metrics["required_found"] == 1.0
        assert outcome.metrics["forbidden_found"] == 0.0

    def test_some_required_missing(self) -> None:
        grader = ContainsGrader("contains", required=["hello", "world", "foo"])
        transcript = _make_transcript("hello world")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["required_found"] == pytest.approx(2.0 / 3.0)

    def test_no_required_empty_list(self) -> None:
        grader = ContainsGrader("contains", required=[])
        transcript = _make_transcript("anything")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0

    def test_forbidden_found_fails(self) -> None:
        grader = ContainsGrader(
            "contains", required=["hello"], forbidden=["secret"],
        )
        transcript = _make_transcript("hello this is secret data")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["forbidden_found"] == 1.0

    def test_forbidden_absent_passes(self) -> None:
        grader = ContainsGrader(
            "contains", required=["hello"], forbidden=["secret"],
        )
        transcript = _make_transcript("hello world")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.metrics["forbidden_found"] == 0.0

    def test_stringifies_non_string_output(self) -> None:
        grader = ContainsGrader("contains", required=["42"])
        transcript = _make_transcript(42)
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True

    def test_multiple_forbidden(self) -> None:
        grader = ContainsGrader(
            "contains", required=[], forbidden=["password", "token"],
        )
        transcript = _make_transcript("your password and token are here")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["forbidden_found"] == 2.0


# ===========================================================================
# RegexMatchGrader
# ===========================================================================

class TestRegexMatchGrader:
    """Tests for RegexMatchGrader."""

    def test_default_policy_is_track(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[r"\d+"])
        assert grader.policy == EvalPolicy.TRACK

    def test_all_patterns_match(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[r"\d+", r"[A-Z]"])
        transcript = _make_transcript("Hello 123")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0
        assert outcome.metrics["patterns_matched"] == 1.0

    def test_some_patterns_match(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[r"\d+", r"[A-Z]", r"@"])
        transcript = _make_transcript("Hello 123")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["patterns_matched"] == pytest.approx(2.0 / 3.0)

    def test_no_patterns_match(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[r"\d+"])
        transcript = _make_transcript("no numbers here")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["patterns_matched"] == 0.0

    def test_empty_patterns_passes(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[])
        transcript = _make_transcript("anything")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0

    def test_stringifies_non_string_output(self) -> None:
        grader = RegexMatchGrader("regex", patterns=[r"42"])
        transcript = _make_transcript(42)
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True

    def test_invalid_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern\\[0\\] is invalid"):
            RegexMatchGrader("regex", patterns=[r"[invalid"])


# ===========================================================================
# ConstraintGrader
# ===========================================================================

class TestConstraintGrader:
    """Tests for ConstraintGrader."""

    def test_default_policy_is_gate(self) -> None:
        grader = ConstraintGrader(
            "constraint", constraints=[{"type": "must_include", "value": "ok"}],
        )
        assert grader.policy == EvalPolicy.GATE

    # -- must_include -------------------------------------------------------

    def test_must_include_present(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[{"type": "must_include", "value": "disclaimer"}],
        )
        transcript = _make_transcript("This includes a disclaimer section.")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.metrics["constraints_met"] == 1.0
        assert outcome.metrics["violations"] == 0.0

    def test_must_include_absent(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[{"type": "must_include", "value": "disclaimer"}],
        )
        transcript = _make_transcript("No special section here.")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["violations"] == 1.0

    # -- must_not_include ---------------------------------------------------

    def test_must_not_include_absent(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[{"type": "must_not_include", "value": "internal_api_key"}],
        )
        transcript = _make_transcript("Public information only.")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True

    def test_must_not_include_present(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[{"type": "must_not_include", "value": "internal_api_key"}],
        )
        transcript = _make_transcript("Use internal_api_key = abc123")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["violations"] == 1.0

    # -- numeric_range ------------------------------------------------------

    def test_numeric_range_within(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )
        transcript = _make_transcript({"confidence": 0.85, "status": "success"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True

    def test_numeric_range_outside(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )
        transcript = _make_transcript({"confidence": 1.5, "status": "success"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False

    def test_numeric_range_missing_field(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )
        transcript = _make_transcript({"status": "success"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["violations"] == 1.0

    def test_numeric_range_non_dict_output(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )
        transcript = _make_transcript("not a dict")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False

    # -- enum ---------------------------------------------------------------

    def test_enum_valid_value(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
            ],
        )
        transcript = _make_transcript({"status": "success"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True

    def test_enum_invalid_value(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "enum", "field": "status", "values": ["success", "partial", "failed"]},
            ],
        )
        transcript = _make_transcript({"status": "unknown"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False

    def test_enum_missing_field(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "enum", "field": "status", "values": ["success"]},
            ],
        )
        transcript = _make_transcript({"other": "data"})
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False

    # -- mixed constraints --------------------------------------------------

    def test_multiple_constraints_all_pass(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "must_include", "value": "result"},
                {"type": "must_not_include", "value": "error"},
                {"type": "numeric_range", "field": "score", "min": 0.0, "max": 1.0},
                {"type": "enum", "field": "status", "values": ["ok", "fail"]},
            ],
        )
        transcript = _make_transcript(
            {"text": "result is good", "score": 0.9, "status": "ok"},
        )
        # must_include/must_not_include check str(output), which includes "result"
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.metrics["constraints_met"] == 1.0
        assert outcome.metrics["violations"] == 0.0

    def test_multiple_constraints_some_fail(self) -> None:
        grader = ConstraintGrader(
            "constraint",
            constraints=[
                {"type": "must_include", "value": "result"},
                {"type": "must_include", "value": "missing_term"},
            ],
        )
        transcript = _make_transcript("result is here")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is False
        assert outcome.metrics["constraints_met"] == pytest.approx(0.5)
        assert outcome.metrics["violations"] == 1.0

    def test_empty_constraints_passes(self) -> None:
        grader = ConstraintGrader("constraint", constraints=[])
        transcript = _make_transcript("anything")
        outcome = _run(grader.grade(transcript, _make_task()))

        assert outcome.passed is True
        assert outcome.score == 1.0

    def test_unknown_constraint_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown type 'unknown_type'"):
            ConstraintGrader(
                "constraint",
                constraints=[{"type": "unknown_type", "value": "x"}],
            )
