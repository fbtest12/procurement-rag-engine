"""Tests for LLM provider factory."""

import os
import pytest

from src.llm.factory import create_llm_provider
from src.llm.base import LLMProvider


class TestLLMFactory:
    def test_create_openai_provider(self):
        provider = create_llm_provider(
            provider="openai", api_key="test-key"
        )
        assert isinstance(provider, LLMProvider)
        assert provider.get_provider_name() == "openai"
        assert provider.get_model_name() == "gpt-4o"

    def test_create_anthropic_provider(self):
        provider = create_llm_provider(
            provider="anthropic", api_key="test-key"
        )
        assert isinstance(provider, LLMProvider)
        assert provider.get_provider_name() == "anthropic"

    def test_custom_model(self):
        provider = create_llm_provider(
            provider="openai", model="gpt-4-turbo", api_key="test-key"
        )
        assert provider.get_model_name() == "gpt-4-turbo"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_provider(provider="unknown", api_key="test")

    def test_case_insensitive(self):
        provider = create_llm_provider(
            provider="OpenAI", api_key="test-key"
        )
        assert provider.get_provider_name() == "openai"

    def test_missing_api_key_raises(self):
        # Temporarily clear env vars
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                create_llm_provider(provider="openai")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
