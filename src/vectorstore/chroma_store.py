"""
ChromaDB vector store implementation.

ChromaDB is used as the default vector store because it:
- Runs embedded (no external service needed for dev/demo)
- Supports persistent storage
- Includes a built-in embedding function (all-MiniLM-L6-v2)
- Can be swapped for a managed service in production

For production at scale, this would be replaced with Pinecone,
Qdrant, or pgvector behind the same VectorStore interface.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.ingestion.chunker import Chunk
from .store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store with persistent storage."""

    def __init__(
        self,
        collection_name: str = "procurement_docs",
        persist_directory: str = "./data/chromadb",
        embedding_model: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Use custom embedding function if specified, otherwise let ChromaDB use its default
        collection_kwargs = {"name": collection_name, "metadata": {"hnsw:space": "cosine"}}
        if embedding_model:
            collection_kwargs["embedding_function"] = self._create_embedding_function(embedding_model)

        self.collection = self.client.get_or_create_collection(**collection_kwargs)

        logger.info(
            f"Initialized ChromaDB collection '{collection_name}' "
            f"at {persist_directory} "
            f"({self.collection.count()} existing documents)"
        )

    def _create_embedding_function(self, model_name: str):
        """Create a sentence-transformers embedding function."""
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    async def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add document chunks to the collection."""
        if not chunks:
            return 0

        # ChromaDB requires unique IDs
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "strategy": chunk.metadata.get("strategy", "unknown"),
            }
            for chunk in chunks
        ]

        # Upsert to handle re-ingestion gracefully
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Upserted {len(chunks)} chunks into '{self.collection_name}'")
        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        Semantic search over the procurement document collection.

        Args:
            query: Natural language query.
            top_k: Maximum results to return.
            score_threshold: Minimum similarity score (0-1, cosine).

        Returns:
            Ranked list of SearchResult objects.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = 1 - distance

            if score_threshold and score < score_threshold:
                continue

            search_results.append(
                SearchResult(
                    content=results["documents"][0][i],
                    score=round(score, 4),
                    chunk_id=results["ids"][0][i],
                    source=results["metadatas"][0][i].get("source", "unknown"),
                    metadata=results["metadatas"][0][i],
                )
            )

        logger.info(
            f"Search '{query[:50]}...' returned {len(search_results)} results"
        )
        return search_results

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source document."""
        # Get IDs matching the source
        results = self.collection.get(
            where={"source": source},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                f"Deleted {len(results['ids'])} chunks from source '{source}'"
            )
            return len(results["ids"])
        return 0

    async def get_collection_stats(self) -> dict:
        """Return statistics about the vector collection."""
        count = self.collection.count()

        # Get unique sources
        all_metadata = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in all_metadata["metadatas"]:
            sources.add(meta.get("source", "unknown"))

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "unique_sources": len(sources),
            "sources": sorted(sources),
            "persist_directory": self.persist_directory,
        }
