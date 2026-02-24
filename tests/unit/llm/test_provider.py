"""Tests for LLM provider abstraction."""

import pytest

from eval_kit.llm.provider import LLMProvider, InMemoryProvider
from eval_kit.llm.factory import create_provider


class TestLLMProviderABC:
    """Tests for LLMProvider abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


class TestInMemoryProvider:
    """Tests for InMemoryProvider (testing helper)."""

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

    def test_unknown_provider_creates_litellm_if_available(self) -> None:
        """When litellm is installed, any model string creates a LiteLLMProvider."""
        from eval_kit.llm.provider import LiteLLMProvider
        provider = create_provider("nonexistent/model")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.model == "nonexistent/model"


class TestLLMGraderWithProvider:
    """Tests for LLMGrader integration with provider."""

    @pytest.mark.asyncio
    async def test_llm_grader_uses_provider(self) -> None:
        from eval_kit.core.grader import LLMGrader
        from eval_kit.core.task import Task
        from eval_kit.core.transcript import Transcript
        from eval_kit.llm.provider import InMemoryProvider

        provider = InMemoryProvider(
            responses=['{"score": 8, "feedback": "Good"}']
        )

        class TestGrader(LLMGrader):
            def build_grading_prompt(self, transcript, task):
                return f"Evaluate: {transcript.final_output}"

            def parse_llm_response(self, response, task):
                import json
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
        from eval_kit.core.grader import LLMGrader
        from eval_kit.core.task import Task
        from eval_kit.core.transcript import Transcript

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
