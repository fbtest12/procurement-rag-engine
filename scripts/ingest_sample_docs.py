"""
Script to ingest sample procurement documents into the vector store.

Usage:
    python -m scripts.ingest_sample_docs
    python -m scripts.ingest_sample_docs --dir ./data/custom_docs
"""

import asyncio
import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import ChunkingStrategy
from src.vectorstore.chroma_store import ChromaVectorStore
from src.api.config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Ingest procurement documents")
    parser.add_argument(
        "--dir",
        default="./data/sample_docs",
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "recursive", "semantic_sections"],
        default="recursive",
        help="Chunking strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    settings = get_settings()

    # Initialize vector store
    store = ChromaVectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

    # Initialize pipeline
    pipeline = IngestionPipeline(
        vector_store=store,
        chunking_strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Ingest
    logger.info(f"Ingesting documents from: {args.dir}")
    stats = await pipeline.ingest_directory(args.dir, recursive=True)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Documents: {stats['total_documents']}")
    logger.info(f"  Chunks:    {stats['total_chunks']}")
    for doc in stats["documents"]:
        logger.info(f"    - {doc['source']} ({doc['word_count']} words)")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
