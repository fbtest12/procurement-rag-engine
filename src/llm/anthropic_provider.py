"""
Anthropic LLM provider implementation.

Supports Claude Sonnet, Opus, and Haiku models via the Anthropic API.
Uses the Messages API with proper system prompt handling.
"""

import os
from typing import Optional

from .base import LLMProvider, LLMResponse

RAG_SYSTEM_PROMPT = """You are a procurement intelligence assistant specializing in government
RFPs, bid documents, and compliance regulations. Answer questions using ONLY the provided
context. If the context doesn't contain enough information to answer, say so explicitly.
Always cite which document section your answer comes from."""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump(),
        )

    async def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        system = system_prompt or RAG_SYSTEM_PROMPT

        context_block = "\n\n---\n\n".join(
            f"[Document Chunk {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""Based on the following retrieved documents, answer the user's question.

RETRIEVED CONTEXT:
{context_block}

USER QUESTION: {query}

Provide a detailed, accurate answer grounded in the context above. Cite specific
document chunks when possible (e.g., "According to Document Chunk 2...")."""

        return await self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_model_name(self) -> str:
        return self.model
