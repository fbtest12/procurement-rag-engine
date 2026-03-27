"""
Core RAG engine — the heart of the system.

Orchestrates the retrieve-then-generate pipeline:
1. Accept a user query
2. Retrieve relevant chunks from the vector store
3. (Optional) Re-rank results for precision
4. Format context + query into an LLM prompt
5. Generate a grounded answer with citations
6. Return structured response with sources

This module also implements query preprocessing and
result postprocessing hooks for extensibility.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.llm.base import LLMProvider, LLMResponse
from src.vectorstore.store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG engine."""

    answer: str
    sources: list[dict]
    query: str
    model: str
    provider: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    chunks_retrieved: int
    chunks_used: int
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "model": self.model,
            "provider": self.provider,
            "performance": {
                "retrieval_time_ms": self.retrieval_time_ms,
                "generation_time_ms": self.generation_time_ms,
                "total_time_ms": self.total_time_ms,
            },
            "retrieval": {
                "chunks_retrieved": self.chunks_retrieved,
                "chunks_used": self.chunks_used,
            },
            "usage": self.usage,
        }


class RAGEngine:
    """
    Production-grade Retrieval-Augmented Generation engine.

    Designed for government procurement document Q&A with:
    - Configurable retrieval parameters
    - Score-based filtering to avoid hallucination
    - Source citation tracking
    - Performance instrumentation
    - Query preprocessing for procurement domain
    """

    # Procurement-specific query expansions
    DOMAIN_SYNONYMS = {
        "rfp": "request for proposal",
        "rfq": "request for quotation",
        "ifb": "invitation for bid",
        "sow": "scope of work",
        "mwbe": "minority women business enterprise",
        "dbe": "disadvantaged business enterprise",
    }

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3,
        max_context_chunks: int = 8,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_chunks = max_context_chunks
        self.system_prompt = system_prompt

    def _preprocess_query(self, query: str) -> str:
        """
        Expand procurement acronyms and normalize query.

        Government procurement has many domain-specific acronyms.
        Expanding them improves retrieval recall.
        """
        expanded = query
        for acronym, expansion in self.DOMAIN_SYNONYMS.items():
            # Case-insensitive replacement that preserves context
            if acronym.lower() in expanded.lower():
                expanded = expanded + f" ({expansion})"
        return expanded

    async def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        temperature: float = 0.0,
    ) -> RAGResponse:
        """
        Execute a full RAG pipeline query.

        Args:
            user_query: The user's natural language question.
            top_k: Override default retrieval count.
            score_threshold: Override default score filter.
            temperature: LLM generation temperature.

        Returns:
            RAGResponse with answer, sources, and performance metrics.
        """
        total_start = time.perf_counter()
        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold

        # Step 1: Preprocess query
        processed_query = self._preprocess_query(user_query)
        logger.info(f"Processed query: '{processed_query}'")

        # Step 2: Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        search_results = await self.vector_store.search(
            query=processed_query,
            top_k=k,
            score_threshold=threshold,
        )
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            f"Retrieved {len(search_results)} chunks "
            f"(threshold={threshold}) in {retrieval_time:.1f}ms"
        )

        # Step 3: Handle no results
        if not search_results:
            total_time = (time.perf_counter() - total_start) * 1000
            return RAGResponse(
                answer=(
                    "I couldn't find relevant information in the procurement "
                    "documents to answer your question. Please try rephrasing "
                    "or check that the relevant documents have been ingested."
                ),
                sources=[],
                query=user_query,
                model=self.llm.get_model_name(),
                provider=self.llm.get_provider_name(),
                retrieval_time_ms=round(retrieval_time, 2),
                generation_time_ms=0,
                total_time_ms=round(total_time, 2),
                chunks_retrieved=0,
                chunks_used=0,
            )

        # Step 4: Select top chunks (respect max context window)
        selected = search_results[: self.max_context_chunks]
        context_chunks = [result.content for result in selected]

        # Step 5: Generate answer
        generation_start = time.perf_counter()
        llm_response: LLMResponse = await self.llm.generate_with_context(
            query=user_query,
            context_chunks=context_chunks,
            system_prompt=self.system_prompt,
            temperature=temperature,
        )
        generation_time = (time.perf_counter() - generation_start) * 1000

        # Step 6: Build source citations
        sources = [
            {
                "source": result.source,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "excerpt": result.content[:200] + "..."
                if len(result.content) > 200
                else result.content,
            }
            for result in selected
        ]

        total_time = (time.perf_counter() - total_start) * 1000

        response = RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=user_query,
            model=llm_response.model,
            provider=llm_response.provider,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2),
            total_time_ms=round(total_time, 2),
            chunks_retrieved=len(search_results),
            chunks_used=len(selected),
            usage=llm_response.usage,
        )

        logger.info(
            f"RAG query completed in {total_time:.1f}ms "
            f"(retrieval={retrieval_time:.1f}ms, "
            f"generation={generation_time:.1f}ms)"
        )

        return response

    async def get_status(self) -> dict:
        """Return engine status and configuration."""
        store_stats = await self.vector_store.get_collection_stats()
        return {
            "llm_provider": self.llm.get_provider_name(),
            "llm_model": self.llm.get_model_name(),
            "retrieval_config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "max_context_chunks": self.max_context_chunks,
            },
            "vector_store": store_stats,
        }
