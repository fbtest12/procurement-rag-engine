"""Tests for document loading."""

import os
import pytest
import tempfile

from src.ingestion.loader import DocumentLoader, Document


@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def sample_txt_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write("REQUEST FOR PROPOSAL\n\nSection 1: Scope of Work\n\nThe vendor shall provide services.")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_dir(sample_txt_file):
    return os.path.dirname(sample_txt_file)


class TestDocumentLoader:
    def test_load_txt_file(self, loader, sample_txt_file):
        doc = loader.load_file(sample_txt_file)
        assert isinstance(doc, Document)
        assert "REQUEST FOR PROPOSAL" in doc.content
        assert doc.doc_type == "txt"
        assert doc.word_count > 0

    def test_doc_id_is_deterministic(self, loader, sample_txt_file):
        doc1 = loader.load_file(sample_txt_file)
        doc2 = loader.load_file(sample_txt_file)
        assert doc1.doc_id == doc2.doc_id

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    def test_unsupported_extension(self, loader):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Unsupported"):
                    loader.load_file(f.name)
            finally:
                os.unlink(f.name)

    def test_load_directory(self, loader, sample_dir):
        docs = loader.load_directory(sample_dir)
        assert len(docs) >= 1

    def test_metadata_present(self, loader, sample_txt_file):
        doc = loader.load_file(sample_txt_file)
        assert "file_path" in doc.metadata
        assert "file_size_bytes" in doc.metadata
        assert doc.metadata["extension"] == ".txt"
