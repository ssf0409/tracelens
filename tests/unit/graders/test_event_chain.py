"""Tests for EventChainVerifier grader."""

import pytest
from pydantic import ValidationError

from tracelens.core.task import Task
from tracelens.core.transcript import StepType, ToolCall, Transcript, TranscriptStep
from tracelens.graders.event_chain import (
    EventChainConfig,
    EventChainVerifier,
    EventExpectation,
    EventMatchType,
    OrderingMode,
)


@pytest.fixture
def task() -> Task:
    return Task(task_id="t1", name="Test", input_data={})


def _make_transcript(steps: list[TranscriptStep]) -> Transcript:
    t = Transcript(task_id="t1")
    for s in steps:
        t.add_step(s)
    return t


def _tool_step(name: str, args: dict | None = None, result: str | None = None) -> TranscriptStep:
    return TranscriptStep(
        step_type=StepType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name=name,
            arguments=args or {},
            result=result,
        ),
    )


def _output_step(content: str) -> TranscriptStep:
    return TranscriptStep(step_type=StepType.AGENT_OUTPUT, content=content)


class TestEventChainAllFound:
    """Tests when all expected events are found."""

    async def test_all_events_found_strict(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="search", match_type=EventMatchType.TOOL_NAME, tool_name="search"),
                EventExpectation(event_id="analyze", match_type=EventMatchType.TOOL_NAME, tool_name="analyze"),
            ],
            ordering=OrderingMode.STRICT,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("search"),
            _tool_step("analyze"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.score == 1.0
        assert outcome.metrics["events_found"] == 2.0
        assert outcome.metrics["events_missing"] == 0.0

    async def test_all_events_found_unordered(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="search", match_type=EventMatchType.TOOL_NAME, tool_name="search"),
                EventExpectation(event_id="analyze", match_type=EventMatchType.TOOL_NAME, tool_name="analyze"),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        # Reversed order — should still pass with UNORDERED
        transcript = _make_transcript([
            _tool_step("analyze"),
            _tool_step("search"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.score == 1.0


class TestEventChainMissing:
    """Tests when some events are missing."""

    async def test_missing_event_require_all(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="search", match_type=EventMatchType.TOOL_NAME, tool_name="search"),
                EventExpectation(event_id="analyze", match_type=EventMatchType.TOOL_NAME, tool_name="analyze"),
            ],
            ordering=OrderingMode.UNORDERED,
            require_all=True,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([_tool_step("search")])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is False
        assert outcome.metrics["events_found"] == 1.0
        assert outcome.metrics["events_missing"] == 1.0

    async def test_partial_scoring(self, task: Task):
        """With require_all=False, partial matches produce proportional scores."""
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="a", match_type=EventMatchType.TOOL_NAME, tool_name="a"),
                EventExpectation(event_id="b", match_type=EventMatchType.TOOL_NAME, tool_name="b"),
                EventExpectation(event_id="c", match_type=EventMatchType.TOOL_NAME, tool_name="c"),
                EventExpectation(event_id="d", match_type=EventMatchType.TOOL_NAME, tool_name="d"),
            ],
            ordering=OrderingMode.UNORDERED,
            require_all=False,
        )
        verifier = EventChainVerifier("ev1", config)

        # Only 2 of 4 found
        transcript = _make_transcript([_tool_step("a"), _tool_step("c")])

        outcome = await verifier.grade(transcript, task)
        assert outcome.score == pytest.approx(0.5)


class TestEventChainOrdering:
    """Tests for ordering enforcement."""

    async def test_strict_wrong_order(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="first", match_type=EventMatchType.TOOL_NAME, tool_name="first"),
                EventExpectation(event_id="second", match_type=EventMatchType.TOOL_NAME, tool_name="second"),
            ],
            ordering=OrderingMode.STRICT,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("second"),
            _tool_step("first"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is False
        # Score penalty: ratio (1.0) × 0.5 for ordering violation
        assert outcome.score == pytest.approx(0.5)

    async def test_partial_ordering_with_after(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="fetch", match_type=EventMatchType.TOOL_NAME, tool_name="fetch"),
                EventExpectation(event_id="parse", match_type=EventMatchType.TOOL_NAME, tool_name="parse", after=["fetch"]),
                EventExpectation(event_id="log", match_type=EventMatchType.TOOL_NAME, tool_name="log"),
            ],
            ordering=OrderingMode.PARTIAL,
        )
        verifier = EventChainVerifier("ev1", config)

        # log before fetch is fine (no constraint), parse after fetch is required
        transcript = _make_transcript([
            _tool_step("log"),
            _tool_step("fetch"),
            _tool_step("parse"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True

    async def test_partial_ordering_violated(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(event_id="fetch", match_type=EventMatchType.TOOL_NAME, tool_name="fetch"),
                EventExpectation(event_id="parse", match_type=EventMatchType.TOOL_NAME, tool_name="parse", after=["fetch"]),
            ],
            ordering=OrderingMode.PARTIAL,
        )
        verifier = EventChainVerifier("ev1", config)

        # parse before fetch violates the `after` constraint
        transcript = _make_transcript([
            _tool_step("parse"),
            _tool_step("fetch"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is False


class TestEventChainMatchTypes:
    """Tests for different match types."""

    async def test_tool_name_and_args(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="search",
                    match_type=EventMatchType.TOOL_NAME_AND_ARGS,
                    tool_name="search",
                    argument_patterns={"query": "python"},
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("search", args={"query": "python", "limit": "10"}),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True

    async def test_tool_name_and_args_mismatch(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="search",
                    match_type=EventMatchType.TOOL_NAME_AND_ARGS,
                    tool_name="search",
                    argument_patterns={"query": "python"},
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("search", args={"query": "javascript"}),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is False

    async def test_regex_arg_matching(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="search",
                    match_type=EventMatchType.TOOL_NAME_AND_ARGS,
                    tool_name="search",
                    argument_patterns={"query": "re:python|javascript"},
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("search", args={"query": "javascript tutorials"}),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True

    async def test_content_regex(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="output",
                    match_type=EventMatchType.CONTENT_REGEX,
                    content_pattern=r"Phase \d+",
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([_output_step("Phase 1: Setup")])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True

    async def test_step_type_match(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="has_output",
                    match_type=EventMatchType.STEP_TYPE,
                    step_type=StepType.AGENT_OUTPUT,
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([_output_step("result")])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True

    async def test_result_regex(self, task: Task):
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="fetch_ok",
                    match_type=EventMatchType.RESULT_REGEX,
                    content_pattern=r"success",
                ),
            ],
            ordering=OrderingMode.UNORDERED,
        )
        verifier = EventChainVerifier("ev1", config)

        transcript = _make_transcript([
            _tool_step("fetch", result="status: success"),
        ])

        outcome = await verifier.grade(transcript, task)
        assert outcome.passed is True


class TestEventExpectationValidation:
    def test_tool_name_match_requires_tool_name(self):
        with pytest.raises(ValidationError, match="requires 'tool_name'"):
            EventExpectation(
                event_id="x",
                match_type=EventMatchType.TOOL_NAME,
            )

    def test_content_regex_requires_content_pattern(self):
        with pytest.raises(ValidationError, match="requires 'content_pattern'"):
            EventExpectation(
                event_id="x",
                match_type=EventMatchType.CONTENT_REGEX,
            )

    def test_step_type_requires_step_type(self):
        with pytest.raises(ValidationError, match="requires 'step_type'"):
            EventExpectation(
                event_id="x",
                match_type=EventMatchType.STEP_TYPE,
            )

    def test_valid_expectation_accepted(self):
        e = EventExpectation(
            event_id="x",
            match_type=EventMatchType.TOOL_NAME,
            tool_name="search",
        )
        assert e.tool_name == "search"
