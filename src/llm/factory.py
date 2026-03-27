"""
Factory for creating LLM provider instances from configuration.

Centralizes provider instantiation so the rest of the application
never needs to import concrete provider classes directly.
"""

from typing import Optional

from .base import LLMProvider


def create_llm_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: One of 'openai', 'anthropic'. Easily extensible.
        model: Model identifier. Uses provider default if not specified.
        api_key: API key. Falls back to environment variables.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from .openai_provider import OpenAIProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIProvider(**kwargs)

    elif provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return AnthropicProvider(**kwargs)

    elif provider == "openrouter":
        from .openrouter_provider import OpenRouterProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return OpenRouterProvider(**kwargs)

    else:
        supported = ["openai", "anthropic", "openrouter"]
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {supported}"
        )
