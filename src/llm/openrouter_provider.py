"""
OpenRouter LLM provider.

OpenRouter provides a unified API to 500+ models (including free tiers)
using an OpenAI-compatible interface. This means we can use the openai
Python SDK with just a different base_url and API key.

Free models available (no credit card required):
- meta-llama/llama-3.3-70b-instruct:free  (best all-around)
- qwen/qwen3-coder-480b:free              (strongest reasoning)
- nvidia/llama-3.1-nemotron-ultra-253b:free (large context)
- openai/gpt-4o-mini:free                  (lightweight)

Rate limits on free tier: ~20 req/min, ~200 req/day.
"""

import os
from typing import Optional

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse

# Default to Llama 3.3 70B — best free model for document Q&A
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RAG_SYSTEM_PROMPT = """You are a procurement document analyst for government agencies.
Answer questions using ONLY the provided context from procurement documents.
If the context doesn't contain enough information, say so clearly.
Always cite which document section your answer comes from.
Be precise, factual, and professional."""


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider using the OpenAI-compatible API.

    Uses the standard openai Python SDK pointed at OpenRouter's
    base URL, so all OpenAI features (streaming, function calling,
    structured outputs) work out of the box.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter. Get a free key at "
                "https://openrouter.ai/keys"
            )

        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self._api_key,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a completion from a single prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or self.model,
            provider=self.get_provider_name(),
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate an answer grounded in retrieved procurement document chunks."""
        # Format context into a numbered block for clear referencing
        formatted_context = "\n\n".join(
            f"[Document Chunk {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        user_message = (
            f"CONTEXT FROM PROCUREMENT DOCUMENTS:\n"
            f"{'=' * 50}\n"
            f"{formatted_context}\n"
            f"{'=' * 50}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer based strictly on the context above. "
            f"Reference specific document chunks in your answer."
        )

        return await self.generate(
            prompt=user_message,
            system_prompt=system_prompt or RAG_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_provider_name(self) -> str:
        return "openrouter"

    def get_model_name(self) -> str:
        return self.model
