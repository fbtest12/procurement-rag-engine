"""
FastAPI application for the Procurement RAG Engine.

Provides REST endpoints for:
- Document ingestion (single file and batch)
- RAG-powered Q&A queries
- System health and collection status
- Document management (list, delete)

Designed for deployment behind a reverse proxy in production.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.llm.factory import create_llm_provider
from src.vectorstore.chroma_store import ChromaVectorStore
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import ChunkingStrategy
from src.rag.engine import RAGEngine
from src.api.config import get_settings

logger = logging.getLogger(__name__)


# ── Request / Response Models ─────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for RAG queries."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="LLM generation temperature")


class QueryResponse(BaseModel):
    """Structured RAG response."""

    answer: str
    sources: list[dict]
    query: str
    model: str
    provider: str
    performance: dict
    retrieval: dict


class IngestRequest(BaseModel):
    """Request body for file ingestion."""

    file_path: str = Field(..., description="Path to file to ingest")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    llm_provider: str
    llm_model: str
    total_chunks: int


# ── Application Factory ──────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Shared state populated during lifespan
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize RAG components on startup."""
        logger.info("Initializing Procurement RAG Engine...")

        # Initialize vector store
        vector_store = ChromaVectorStore(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_persist_dir,
        )

        # Initialize LLM provider
        llm = create_llm_provider(
            provider=settings.llm_provider,
            model=settings.llm_model,
        )

        # Initialize RAG engine
        rag_engine = RAGEngine(
            llm_provider=llm,
            vector_store=vector_store,
            top_k=settings.default_top_k,
            score_threshold=settings.default_score_threshold,
        )

        # Initialize ingestion pipeline
        ingestion_pipeline = IngestionPipeline(
            vector_store=vector_store,
            chunking_strategy=ChunkingStrategy(settings.chunking_strategy),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        state["rag_engine"] = rag_engine
        state["ingestion_pipeline"] = ingestion_pipeline
        state["vector_store"] = vector_store

        logger.info(
            f"RAG Engine ready: provider={settings.llm_provider}, "
            f"model={settings.llm_model}"
        )

        yield

        logger.info("Shutting down Procurement RAG Engine")

    app = FastAPI(
        title="Procurement RAG Engine",
        description=(
            "AI-powered Q&A system for government procurement documents. "
            "Supports RFPs, bid documents, compliance regulations, and "
            "policy memos with source-cited answers."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Endpoints ────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """System health and status."""
        engine: RAGEngine = state["rag_engine"]
        status = await engine.get_status()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            llm_provider=status["llm_provider"],
            llm_model=status["llm_model"],
            total_chunks=status["vector_store"]["total_chunks"],
        )

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """
        Ask a question about ingested procurement documents.

        The system retrieves relevant document chunks and generates
        a grounded answer with source citations.
        """
        engine: RAGEngine = state["rag_engine"]

        try:
            result = await engine.query(
                user_query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                temperature=request.temperature,
            )

            return QueryResponse(
                answer=result.answer,
                sources=result.sources,
                query=result.query,
                model=result.model,
                provider=result.provider,
                performance={
                    "retrieval_time_ms": result.retrieval_time_ms,
                    "generation_time_ms": result.generation_time_ms,
                    "total_time_ms": result.total_time_ms,
                },
                retrieval={
                    "chunks_retrieved": result.chunks_retrieved,
                    "chunks_used": result.chunks_used,
                },
            )
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest/file")
    async def ingest_file(request: IngestRequest):
        """Ingest a single procurement document."""
        pipeline: IngestionPipeline = state["ingestion_pipeline"]

        try:
            stats = await pipeline.ingest_file(request.file_path)
            return {"status": "success", "stats": stats}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest/directory")
    async def ingest_directory(dir_path: str, recursive: bool = False):
        """Ingest all supported documents in a directory."""
        pipeline: IngestionPipeline = state["ingestion_pipeline"]

        try:
            stats = await pipeline.ingest_directory(dir_path, recursive=recursive)
            return {"status": "success", "stats": stats}
        except NotADirectoryError:
            raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/documents")
    async def list_documents():
        """List all ingested document sources and collection stats."""
        store: ChromaVectorStore = state["vector_store"]
        stats = await store.get_collection_stats()
        return stats

    @app.delete("/documents/{source}")
    async def delete_document(source: str):
        """Remove all chunks from a specific source document."""
        store: ChromaVectorStore = state["vector_store"]
        deleted = await store.delete_by_source(source)
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: {source}",
            )
        return {"status": "deleted", "source": source, "chunks_removed": deleted}

    return app
