"""
Document ingestion module.
Parses PDF files via pdfplumber or accepts raw text, then splits
the content into paragraph-level chunks for downstream processing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from models.schemas import DocumentChunk

# Maximum characters per chunk before a forced split
_CHUNK_MAX_CHARS: int = 1_200
# Minimum characters to keep a chunk (discard boilerplate whitespace chunks)
_CHUNK_MIN_CHARS: int = 30


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples from a PDF file."""
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines; further split long paragraphs."""
    raw_paras = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for para in raw_paras:
        para = para.strip()
        if len(para) <= _CHUNK_MAX_CHARS:
            paragraphs.append(para)
        else:
            # Hard-split at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > _CHUNK_MAX_CHARS and current:
                    paragraphs.append(" ".join(current))
                    current, current_len = [], 0
                current.append(sent)
                current_len += len(sent) + 1
            if current:
                paragraphs.append(" ".join(current))
    return [p for p in paragraphs if len(p) >= _CHUNK_MIN_CHARS]


def _build_chunks(
    paragraphs: list[str], source_page: int | None = None
) -> list[DocumentChunk]:
    """Convert a list of paragraph strings into DocumentChunk objects."""
    chunks: list[DocumentChunk] = []
    char_cursor = 0
    for idx, para in enumerate(paragraphs):
        chunks.append(
            DocumentChunk(
                chunk_index=idx,
                text=para,
                source_page=source_page,
                char_start=char_cursor,
                char_end=char_cursor + len(para),
            )
        )
        char_cursor += len(para) + 2  # +2 for the blank-line separator
    return chunks


def ingest_pdf(path: str | Path) -> list[DocumentChunk]:
    """Parse a PDF and return a flat list of DocumentChunks across all pages."""
    pages = load_pdf(path)
    chunks: list[DocumentChunk] = []
    global_idx = 0
    for page_num, page_text in pages:
        paragraphs = _split_into_paragraphs(page_text)
        for para in paragraphs:
            if len(para) < _CHUNK_MIN_CHARS:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_index=global_idx,
                    text=para,
                    source_page=page_num,
                )
            )
            global_idx += 1
    return chunks


def ingest_text(text: str) -> list[DocumentChunk]:
    """Parse a plain-text string and return a list of DocumentChunks."""
    paragraphs = _split_into_paragraphs(text)
    return _build_chunks(paragraphs, source_page=None)


def ingest(source: str | Path) -> list[DocumentChunk]:
    """
    Unified entry point.
    Accepts a file path (str/Path) or a raw text string.
    Auto-detects PDF vs plain text.
    """
    if isinstance(source, Path) or (
        isinstance(source, str) and Path(source).suffix.lower() == ".pdf"
    ):
        path = Path(source)
        if path.exists() and path.suffix.lower() == ".pdf":
            return ingest_pdf(path)
    # Treat as raw text
    return ingest_text(str(source))


def full_text(chunks: list[DocumentChunk]) -> str:
    """Reconstruct a single string from all chunks (for LLM context windows)."""
    return "\n\n".join(c.text for c in chunks)
