"""Tests for LLM provider abstraction."""

import json

import pytest

from tracelens.core.grader import LLMGrader
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.llm.factory import create_provider
from tracelens.llm.provider import InMemoryProvider, LLMProvider


class TestLLMProviderABC:
    """Tests for LLMProvider abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


class TestInMemoryProvider:
    """Tests for InMemoryProvider (testing helper)."""

    def test_empty_responses_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one response"):
            InMemoryProvider(responses=[])

    @pytest.mark.asyncio
    async def test_returns_canned_response(self) -> None:
        provider = InMemoryProvider(responses=["hello world"])
        result = await provider.complete("test prompt")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_cycles_through_responses(self) -> None:
        provider = InMemoryProvider(responses=["first", "second"])
        assert await provider.complete("p1") == "first"
        assert await provider.complete("p2") == "second"
        assert await provider.complete("p3") == "first"  # wraps around

    @pytest.mark.asyncio
    async def test_records_prompts(self) -> None:
        provider = InMemoryProvider(responses=["ok"])
        await provider.complete("prompt 1")
        await provider.complete("prompt 2")
        assert provider.prompts == ["prompt 1", "prompt 2"]


class TestFactory:
    """Tests for create_provider factory."""

    def test_create_in_memory_provider(self) -> None:
        provider = create_provider("in-memory", responses=["test"])
        assert isinstance(provider, InMemoryProvider)

    def test_non_in_memory_alias_raises_with_guidance(self) -> None:
        """tracelens no longer ships a built-in third-party provider
        wrapper. Calling the factory with any non-'in-memory' alias
        raises ValueError and points at the subclassing pattern."""
        with pytest.raises(ValueError, match="subclass LLMProvider"):
            create_provider("anthropic/claude-3-opus")


class TestLLMGraderWithProvider:
    """Tests for LLMGrader integration with provider."""

    @pytest.mark.asyncio
    async def test_llm_grader_uses_provider(self) -> None:
        provider = InMemoryProvider(
            responses=['{"score": 8, "feedback": "Good"}']
        )

        class TestGrader(LLMGrader):
            def build_grading_prompt(self, transcript, task):
                return f"Evaluate: {transcript.final_output}"

            def parse_llm_response(self, response, task):
                data = json.loads(response)
                score = data["score"] / 10.0
                return score >= 0.7, score, {}, data["feedback"]

        grader = TestGrader("test-llm", provider=provider)
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(task_id="t1", final_output="test output")

        outcome = await grader.grade(transcript, task)
        assert outcome.passed is True
        assert outcome.score == pytest.approx(0.8)
        assert provider.prompts == ["Evaluate: test output"]

    @pytest.mark.asyncio
    async def test_llm_grader_without_provider_raises(self) -> None:
        """Without a provider, _call_llm still raises NotImplementedError."""

        class BareGrader(LLMGrader):
            def build_grading_prompt(self, transcript, task):
                return "prompt"

            def parse_llm_response(self, response, task):
                return True, 1.0, {}, ""

        grader = BareGrader("bare")
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(task_id="t1")

        with pytest.raises(NotImplementedError):
            await grader.grade(transcript, task)


class TestLLMGraderParseFailure:
    """Test that LLMGrader wraps parse_llm_response failures with context."""

    @pytest.mark.asyncio
    async def test_parse_failure_includes_grader_id_and_preview(self) -> None:
        provider = InMemoryProvider(responses=["not valid json at all"])

        class BadParseGrader(LLMGrader):
            def build_grading_prompt(self, transcript, task):
                return "evaluate this"

            def parse_llm_response(self, response, task):
                return json.loads(response)  # will raise JSONDecodeError

        grader = BadParseGrader("my-grader", provider=provider)
        task = Task(task_id="t1", name="Test", input_data={})
        transcript = Transcript(task_id="t1", final_output="output")

        with pytest.raises(RuntimeError, match="my-grader") as exc_info:
            await grader.grade(transcript, task)

        error_msg = str(exc_info.value)
        assert "not valid json" in error_msg
        assert "parse_llm_response failed" in error_msg
