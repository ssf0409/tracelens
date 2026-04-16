"""Tests for LLM provider abstraction."""

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from eval_kit.core.grader import LLMGrader
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript
from eval_kit.llm.factory import create_provider
from eval_kit.llm.provider import InMemoryProvider, LLMProvider

# `litellm` is an optional dependency (install with `pip install "eval-kit[llm]"`).
# Tests that import LiteLLMProvider must be skipped when it's absent so the
# core-only CI job stays green.
requires_litellm = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None,
    reason="requires 'litellm' (install with: pip install 'eval-kit[llm]')",
)


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

    @requires_litellm
    def test_unknown_provider_creates_litellm_if_available(self) -> None:
        """When litellm is installed, any model string creates a LiteLLMProvider."""
        from eval_kit.llm.litellm_provider import LiteLLMProvider

        provider = create_provider("nonexistent/model")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.model == "nonexistent/model"


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


@requires_litellm
class TestLiteLLMProvider:
    """Behavioral tests for LiteLLMProvider using mocked litellm."""

    @pytest.mark.asyncio
    async def test_wraps_api_errors_as_runtime_error(self) -> None:
        from eval_kit.llm.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(model="test/model")
        with patch(
            "eval_kit.llm.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=ConnectionError("network down"),
        ):
            with pytest.raises(RuntimeError, match="test/model"):
                await provider.complete("hello")

    @pytest.mark.asyncio
    async def test_raises_on_null_content(self) -> None:
        from eval_kit.llm.litellm_provider import LiteLLMProvider

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        provider = LiteLLMProvider(model="test/model")
        with patch(
            "eval_kit.llm.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError, match="null content"):
                await provider.complete("hello")

    @pytest.mark.asyncio
    async def test_merges_default_and_call_kwargs(self) -> None:
        from eval_kit.llm.litellm_provider import LiteLLMProvider

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        mock_acompletion = AsyncMock(return_value=mock_response)
        provider = LiteLLMProvider(model="test/model", temperature=0.5)

        with patch(
            "eval_kit.llm.litellm_provider.litellm.acompletion",
            mock_acompletion,
        ):
            result = await provider.complete("hello", max_tokens=100)

        assert result == "ok"
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_call_kwargs_override_defaults(self) -> None:
        from eval_kit.llm.litellm_provider import LiteLLMProvider

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        mock_acompletion = AsyncMock(return_value=mock_response)
        provider = LiteLLMProvider(model="test/model", temperature=0.5)

        with patch(
            "eval_kit.llm.litellm_provider.litellm.acompletion",
            mock_acompletion,
        ):
            await provider.complete("hello", temperature=0.9)

        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.9


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
