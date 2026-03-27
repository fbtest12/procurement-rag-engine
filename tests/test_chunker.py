"""Tests for document chunking strategies."""

import pytest
from src.ingestion.loader import Document
from src.ingestion.chunker import DocumentChunker, ChunkingStrategy


@pytest.fixture
def sample_document():
    return Document(
        content=(
            "SECTION 1: SCOPE OF WORK\n\n"
            "The contractor shall provide IT managed services including "
            "24/7 help desk support, network monitoring, and cybersecurity "
            "operations for all City departments.\n\n"
            "SECTION 2: EVALUATION CRITERIA\n\n"
            "Proposals will be scored on technical approach (35 points), "
            "qualifications (25 points), staffing (15 points), and "
            "cost (20 points). MBE/WBE participation accounts for 5 points.\n\n"
            "SECTION 3: INSURANCE REQUIREMENTS\n\n"
            "The vendor must maintain Commercial General Liability of "
            "$2,000,000 per occurrence and Cyber Liability of $10,000,000."
        ),
        source="test_rfp.txt",
        doc_type="txt",
    )


class TestFixedSizeChunking:
    def test_produces_chunks(self, sample_document):
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=200,
            chunk_overlap=50,
        )
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 1

    def test_chunks_have_metadata(self, sample_document):
        chunker = DocumentChunker(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=200)
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.source == "test_rfp.txt"
            assert chunk.doc_id == sample_document.doc_id
            assert "strategy" in chunk.metadata

    def test_chunk_ids_are_unique(self, sample_document):
        chunker = DocumentChunker(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=200)
        chunks = chunker.chunk_document(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestRecursiveChunking:
    def test_respects_paragraph_boundaries(self, sample_document):
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300,
            chunk_overlap=0,
        )
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) >= 2

    def test_small_document_single_chunk(self):
        doc = Document(content="Short text.", source="small.txt", doc_type="txt")
        chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=1000)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1


class TestSemanticChunking:
    def test_splits_on_section_headers(self, sample_document):
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC_SECTIONS,
            chunk_size=500,
        )
        chunks = chunker.chunk_document(sample_document)
        # Should detect the 3 SECTION headers
        assert len(chunks) >= 3

    def test_fallback_for_unstructured_docs(self):
        doc = Document(
            content="Just a plain paragraph without any section markers. " * 20,
            source="plain.txt",
            doc_type="txt",
        )
        chunker = DocumentChunker(strategy=ChunkingStrategy.SEMANTIC_SECTIONS, chunk_size=200)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1


class TestBatchChunking:
    def test_chunk_multiple_documents(self, sample_document):
        doc2 = Document(content="Another document.", source="doc2.txt", doc_type="txt")
        chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=200)
        chunks = chunker.chunk_documents([sample_document, doc2])
        sources = {c.source for c in chunks}
        assert "test_rfp.txt" in sources
        assert "doc2.txt" in sources
