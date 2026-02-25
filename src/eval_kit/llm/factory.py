"""Factory for creating LLM providers."""

from typing import Any

from eval_kit.llm.litellm_provider import LiteLLMProvider
from eval_kit.llm.provider import InMemoryProvider, LLMProvider


def create_provider(model_or_alias: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider from a model string or alias.

    Args:
        model_or_alias: Either "in-memory" for testing, or a LiteLLM model
            string like "anthropic/claude-3-opus", "openai/gpt-4", etc.
        **kwargs: Additional arguments passed to the provider.

    Returns:
        An LLMProvider instance.
    """
    if model_or_alias == "in-memory":
        return InMemoryProvider(responses=kwargs.pop("responses", ["mock response"]))

    return LiteLLMProvider(model=model_or_alias, **kwargs)
