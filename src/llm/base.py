"""
Abstract base class for LLM providers.

This module defines the contract that all LLM providers must implement,
enabling seamless swapping between OpenAI, Anthropic, Azure, local models,
or any future provider without touching downstream code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response envelope from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    raw_response: Optional[dict] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """
    Abstract interface for language model providers.

    Implementations must handle:
    - Authentication and client initialization
    - Message formatting per provider spec
    - Streaming and non-streaming completions
    - Error handling and retries
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a completion from a single prompt string."""
        ...

    @abstractmethod
    async def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate a response grounded in retrieved context chunks.

        This is the primary method used by the RAG pipeline — it formats
        the retrieved documents into the prompt and asks the LLM to
        synthesize an answer.
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider identifier (e.g. 'openai', 'anthropic')."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier (e.g. 'gpt-4o', 'claude-sonnet-4-20250514')."""
        ...
