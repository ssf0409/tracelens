"""Tests for LLMGrader retry and timeout behavior.

GraderConfig declares timeout_seconds / retry_on_error / max_retries;
these tests pin down that LLMGrader.grade() actually honors them.
"""

import asyncio
import json

import pytest

from tracelens.core.grader import GraderConfig, LLMGrader
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript


class _FlakyProvider:
    """Provider that fails N times before succeeding."""

    def __init__(self, failures: int, response: str = '{"score": 0.8}'):
        self.failures = failures
        self.response = response
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError(f"transient failure #{self.calls}")
        return self.response


class _HangingProvider:
    """Provider that never returns within any reasonable timeout."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(60)
        return "{}"


class _SequenceProvider:
    """Provider that returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class _JSONGrader(LLMGrader):
    """Minimal LLM grader parsing a JSON score."""

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        return "grade it"

    def parse_llm_response(
        self, response: str, task: Task
    ) -> tuple[bool, float, dict[str, float], str]:
        data = json.loads(response)
        score = float(data["score"])
        return score >= 0.5, score, {}, "ok"


def _config(**overrides: object) -> GraderConfig:
    defaults: dict[str, object] = {"retry_backoff_seconds": 0.0}
    defaults.update(overrides)
    return GraderConfig(**defaults)


def _task() -> Task:
    return Task(task_id="t1", name="t", description="d", input_data={"x": 1})


def _transcript() -> Transcript:
    return Transcript(task_id="t1")


def test_retries_transient_call_failures() -> None:
    provider = _FlakyProvider(failures=2)
    grader = _JSONGrader("g", provider=provider, config=_config(max_retries=3))

    outcome = asyncio.run(grader.grade(_transcript(), _task()))

    assert provider.calls == 3
    assert outcome.passed is True
    assert outcome.score == 0.8


def test_no_retry_when_disabled() -> None:
    provider = _FlakyProvider(failures=1)
    grader = _JSONGrader(
        "g", provider=provider, config=_config(retry_on_error=False)
    )

    with pytest.raises(ConnectionError):
        asyncio.run(grader.grade(_transcript(), _task()))
    assert provider.calls == 1


def test_retry_exhaustion_raises_last_error() -> None:
    provider = _FlakyProvider(failures=100)
    grader = _JSONGrader("g", provider=provider, config=_config(max_retries=2))

    with pytest.raises(ConnectionError, match="transient failure #3"):
        asyncio.run(grader.grade(_transcript(), _task()))
    assert provider.calls == 3  # 1 initial + 2 retries


def test_timeout_enforced_on_llm_call() -> None:
    provider = _HangingProvider()
    grader = _JSONGrader(
        "g",
        provider=provider,
        config=_config(timeout_seconds=0.05, max_retries=1),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(grader.grade(_transcript(), _task()))
    assert provider.calls == 2  # timeout is retried like any other error


def test_retries_on_malformed_response() -> None:
    provider = _SequenceProvider(["not json at all", '{"score": 0.9}'])
    grader = _JSONGrader("g", provider=provider, config=_config(max_retries=2))

    outcome = asyncio.run(grader.grade(_transcript(), _task()))

    assert provider.calls == 2
    assert outcome.passed is True
    assert outcome.score == 0.9


def test_malformed_response_exhaustion_includes_preview() -> None:
    provider = _SequenceProvider(["garbage"])
    grader = _JSONGrader("g", provider=provider, config=_config(max_retries=1))

    with pytest.raises(RuntimeError, match="garbage"):
        asyncio.run(grader.grade(_transcript(), _task()))
    assert provider.calls == 2


def test_missing_provider_raises_immediately_without_retry() -> None:
    grader = _JSONGrader("g", config=_config(max_retries=3))

    with pytest.raises(NotImplementedError):
        asyncio.run(grader.grade(_transcript(), _task()))
