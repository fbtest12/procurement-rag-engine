"""
Abstract vector store interface.

Defines the contract for vector storage backends.
ChromaDB is the default implementation, but the interface
supports swapping to Pinecone, Weaviate, Qdrant, pgvector, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    """A single vector search result with score and metadata."""

    content: str
    score: float
    chunk_id: str
    source: str
    metadata: dict

    @property
    def is_relevant(self) -> bool:
        """Heuristic relevance check (tunable threshold)."""
        return self.score >= 0.7


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add document chunks. Returns count of chunks added."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """Semantic search. Returns ranked results."""
        ...

    @abstractmethod
    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source document."""
        ...

    @abstractmethod
    async def get_collection_stats(self) -> dict:
        """Return stats about the vector collection."""
        ...
