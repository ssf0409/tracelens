"""Factory for creating LLM providers."""

from typing import Any

from eval_kit.llm.provider import InMemoryProvider, LLMProvider


def create_provider(model_or_alias: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider from a model string or alias.

    Args:
        model_or_alias: Either "in-memory" for testing, or a LiteLLM model
            string like "anthropic/claude-3-opus", "openai/gpt-4", etc.
        **kwargs: Additional arguments passed to the provider.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider type is unknown or litellm is not installed.
    """
    if model_or_alias == "in-memory":
        return InMemoryProvider(responses=kwargs.pop("responses", ["mock response"]))

    # Try LiteLLM for all other model strings
    try:
        from eval_kit.llm.provider import LiteLLMProvider
        return LiteLLMProvider(model=model_or_alias, **kwargs)
    except ImportError:
        raise ValueError(
            f"Unknown provider '{model_or_alias}'. "
            f"Install litellm (`pip install eval-kit[llm]`) for LLM provider support."
        )
