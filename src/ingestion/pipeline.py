"""
End-to-end ingestion pipeline.

Orchestrates loading, chunking, and embedding of procurement documents
into the vector store. Designed to be run as a batch job or triggered
via the API for incremental document addition.
"""

import logging
from typing import Optional

from .loader import DocumentLoader, Document
from .chunker import DocumentChunker, Chunk, ChunkingStrategy

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates document loading → chunking → vector store insertion.

    Usage:
        pipeline = IngestionPipeline(vector_store=store)
        stats = await pipeline.ingest_directory("./data/rfps/")
    """

    def __init__(
        self,
        vector_store=None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = vector_store

    async def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file into the vector store."""
        document = self.loader.load_file(file_path)
        chunks = self.chunker.chunk_document(document)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)

        stats = {
            "source": document.source,
            "doc_id": document.doc_id,
            "word_count": document.word_count,
            "chunk_count": len(chunks),
            "avg_chunk_size": (
                sum(len(c.content) for c in chunks) // len(chunks)
                if chunks
                else 0
            ),
        }
        logger.info(f"Ingested {document.source}: {stats}")
        return stats

    async def ingest_directory(
        self, dir_path: str, recursive: bool = False
    ) -> dict:
        """Ingest all supported files in a directory."""
        documents = self.loader.load_directory(dir_path, recursive=recursive)
        all_chunks = self.chunker.chunk_documents(documents)

        if self.vector_store:
            await self.vector_store.add_chunks(all_chunks)

        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "documents": [
                {
                    "source": doc.source,
                    "doc_id": doc.doc_id,
                    "word_count": doc.word_count,
                }
                for doc in documents
            ],
        }
        logger.info(
            f"Ingested directory {dir_path}: "
            f"{len(documents)} docs, {len(all_chunks)} chunks"
        )
        return stats
