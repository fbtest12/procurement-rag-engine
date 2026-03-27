"""
OpenAI LLM provider implementation.

Supports GPT-4o, GPT-4-turbo, and other OpenAI chat models.
Handles authentication via API key and provides standardized
responses through the LLMProvider interface.
"""

import os
from typing import Optional

from .base import LLMProvider, LLMResponse

RAG_SYSTEM_PROMPT = """You are a procurement intelligence assistant specializing in government
RFPs, bid documents, and compliance regulations. Answer questions using ONLY the provided
context. If the context doesn't contain enough information to answer, say so explicitly.
Always cite which document section your answer comes from."""


class OpenAIProvider(LLMProvider):
    """OpenAI API provider with automatic client management."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=self.model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
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
        return "openai"

    def get_model_name(self) -> str:
        return self.model
