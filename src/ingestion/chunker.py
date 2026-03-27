"""
Document chunking strategies for RAG.

Implements multiple chunking approaches optimized for procurement
documents, which often have section headers, numbered clauses,
and table-heavy content.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re
import logging

from .loader import Document

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC_SECTIONS = "semantic_sections"


@dataclass
class Chunk:
    """A chunk of text with provenance metadata."""

    content: str
    doc_id: str
    chunk_index: int
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_chunk_{self.chunk_index}"

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English)."""
        return len(self.content) // 4


class DocumentChunker:
    """
    Configurable document chunker with multiple strategies.

    The choice of chunking strategy significantly impacts RAG quality.
    Procurement documents benefit from semantic section splitting
    because they have clear structural markers (numbered sections,
    headers like "SCOPE OF WORK", "TERMS AND CONDITIONS", etc.).
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using the configured strategy."""
        strategy_map = {
            ChunkingStrategy.FIXED_SIZE: self._fixed_size_chunk,
            ChunkingStrategy.RECURSIVE: self._recursive_chunk,
            ChunkingStrategy.SEMANTIC_SECTIONS: self._semantic_chunk,
        }

        chunker = strategy_map[self.strategy]
        chunks = chunker(document)

        logger.info(
            f"Chunked '{document.source}' into {len(chunks)} chunks "
            f"(strategy={self.strategy.value})"
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _fixed_size_chunk(self, document: Document) -> list[Chunk]:
        """Simple character-based splitting with overlap."""
        text = document.content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                boundary = text.rfind(". ", start + self.chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        source=document.source,
                        metadata={
                            "strategy": "fixed_size",
                            "char_start": start,
                            "char_end": end,
                            **document.metadata,
                        },
                    )
                )
                chunk_index += 1

            start = end - self.chunk_overlap

        return chunks

    def _recursive_chunk(self, document: Document) -> list[Chunk]:
        """
        Recursively split using hierarchical separators.

        Tries to split on section breaks first, then paragraphs,
        then sentences, preserving document structure.
        """
        separators = [
            "\n\n\n",  # Major section breaks
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            " ",  # Words (last resort)
        ]

        raw_chunks = self._recursive_split(
            document.content, separators, self.chunk_size
        )

        chunks = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if text:
                chunks.append(
                    Chunk(
                        content=text,
                        doc_id=document.doc_id,
                        chunk_index=i,
                        source=document.source,
                        metadata={
                            "strategy": "recursive",
                            **document.metadata,
                        },
                    )
                )

        # Apply overlap by prepending tail of previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_text = chunks[i - 1].content
                overlap_text = prev_text[-self.chunk_overlap :]
                chunks[i].content = overlap_text + " " + chunks[i].content

        return chunks

    def _recursive_split(
        self, text: str, separators: list[str], max_size: int
    ) -> list[str]:
        """Recursively split text using a hierarchy of separators."""
        if len(text) <= max_size:
            return [text]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                result = []
                current = ""
                for part in parts:
                    candidate = current + sep + part if current else part
                    if len(candidate) <= max_size:
                        current = candidate
                    else:
                        if current:
                            result.append(current)
                        if len(part) > max_size:
                            remaining_seps = separators[
                                separators.index(sep) + 1 :
                            ]
                            if remaining_seps:
                                result.extend(
                                    self._recursive_split(
                                        part, remaining_seps, max_size
                                    )
                                )
                            else:
                                result.append(part[:max_size])
                        else:
                            current = part
                if current:
                    result.append(current)
                return result

        # Fallback: hard split
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    def _semantic_chunk(self, document: Document) -> list[Chunk]:
        """
        Split on semantic section boundaries common in procurement docs.

        Looks for patterns like:
        - "SECTION 1: SCOPE OF WORK"
        - "1.0 General Requirements"
        - "Article IV - Terms and Conditions"
        """
        section_pattern = re.compile(
            r"(?:^|\n)"
            r"(?:"
            r"(?:SECTION|ARTICLE|PART)\s+[\dIVXivx]+[.:)\s-]"
            r"|[\d]+\.[\d]*\s+[A-Z]"
            r"|[A-Z][A-Z\s]{5,}(?:\n|$)"
            r")",
            re.MULTILINE,
        )

        splits = list(section_pattern.finditer(document.content))

        if len(splits) < 2:
            # Not enough section markers — fall back to recursive
            return self._recursive_chunk(document)

        sections = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(document.content)
            section_text = document.content[start:end].strip()
            if section_text:
                sections.append(section_text)

        # Sub-chunk any oversized sections
        chunks = []
        chunk_index = 0
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=section,
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        source=document.source,
                        metadata={
                            "strategy": "semantic_sections",
                            **document.metadata,
                        },
                    )
                )
                chunk_index += 1
            else:
                # Sub-chunk using recursive strategy
                sub_doc = Document(
                    content=section,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata=document.metadata,
                )
                for sub_chunk in self._recursive_chunk(sub_doc):
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.metadata["strategy"] = "semantic_sections"
                    chunks.append(sub_chunk)
                    chunk_index += 1

        return chunks
