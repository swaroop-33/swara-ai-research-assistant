"""
app/document_processor.py
==========================
Pure document processing layer — extraction, normalization, and chunking.

Architectural role:
  - This module is a stateless utility layer.
  - It receives file paths or raw text; it returns structured data.
  - It has ZERO knowledge of FastAPI, services, databases, or embeddings.
  - This makes it independently testable and reusable.

Why this separation matters:
  - Ingestion quality directly determines retrieval quality.
  - Chunking with overlap ensures semantic continuity across boundaries.
  - Rich per-chunk metadata enables source attribution in the UI.
"""

import re
import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pypdf

import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single text chunk enriched with source metadata.

    Every field here maps directly to the RetrievedChunk schema defined in
    app/schemas/response.py — this is the data contract between ingestion
    and the retrieval transparency UI.
    """

    chunk_id: str           # Deterministic: "{stem}_chunk_{index:04d}"
    text: str               # The actual chunk text
    filename: str           # Original uploaded filename
    document_type: str      # "pdf" or "txt"
    page_number: Optional[int]  # Page where chunk starts (None for txt)
    chunk_index: int        # 0-based position in document
    char_count: int         # len(text) — for analytics
    timestamp: str          # ISO 8601 ingestion time (UTC)


@dataclass
class ExtractionResult:
    """
    Intermediate result from text extraction (before chunking).
    Carries page-level text with positions for accurate page attribution.
    """

    pages: List[Tuple[int, str]]  # List of (page_number, page_text)
    total_characters: int
    total_pages: int
    document_type: str


# ─────────────────────────────────────────────
# Text Normalization
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Clean raw extracted text for consistent chunking quality.

    Operations (in order):
      1. Strip null bytes — common artifact in PDF extraction
      2. Normalize line endings to Unix-style
      3. Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
      4. Normalize tab characters to single spaces
      5. Collapse multiple spaces to single space per line
      6. Strip leading/trailing whitespace from each line
      7. Final strip of the whole string

    Why normalize before chunking?
      Inconsistent whitespace inflates chunk size counters, creates chunks
      that are mostly whitespace, and degrades embedding quality by adding
      noise tokens.
    """
    if not text:
        return ""

    # 1. Remove null bytes
    text = text.replace("\x00", "")

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse excessive blank lines (3+ newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Normalize tabs to spaces
    text = text.replace("\t", " ")

    # 5-6. Collapse multiple spaces and strip each line
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 7. Final strip
    return text.strip()


# ─────────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_path: str | Path) -> ExtractionResult:
    """
    Extract text from PDF using PyMuPDF (fitz).

    Why PyMuPDF over pypdf?
      - Better spacing reconstruction
      - Better screenplay/document layout extraction
      - Faster extraction
      - Better Unicode handling
      - Better paragraph continuity
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(
        f"Starting PDF extraction with PyMuPDF | file={file_path.name}"
    )

    pages: List[Tuple[int, str]] = []
    total_chars = 0
    failed_pages = 0

    try:

        doc = fitz.open(str(file_path))

        logger.debug(
            f"PDF opened | pages={len(doc)}"
        )

        for page_index in range(len(doc)):

            page_number = page_index + 1

            try:

                page = doc.load_page(page_index)

                # Better extraction mode
                raw_text = page.get_text("text")

                cleaned = normalize_text(raw_text)

                if not cleaned.strip():

                    logger.warning(
                        f"Empty page extracted | "
                        f"file={file_path.name} | "
                        f"page={page_number}"
                    )

                    continue

                pages.append(
                    (page_number, cleaned)
                )

                total_chars += len(cleaned)

                logger.debug(
                    f"Page extracted | "
                    f"page={page_number} | "
                    f"chars={len(cleaned)}"
                )

            except Exception as e:

                failed_pages += 1

                logger.warning(
                    f"Page extraction failed | "
                    f"page={page_number} | "
                    f"error={e}"
                )

                continue

        doc.close()

    except Exception as e:

        raise ValueError(
            f"Failed to read PDF '{file_path.name}': {e}"
        ) from e

    if not pages:

        raise ValueError(
            f"No text extracted from '{file_path.name}'."
        )

    logger.info(
        f"PDF extraction complete | "
        f"file={file_path.name} | "
        f"pages={len(pages)} | "
        f"failed={failed_pages} | "
        f"chars={total_chars}"
    )

    return ExtractionResult(
        pages=pages,
        total_characters=total_chars,
        total_pages=len(pages),
        document_type="pdf",
    )

def extract_text_from_txt(file_path: str | Path) -> ExtractionResult:
    """
    Extract and normalize text from a .txt file.

    Plain text has no page concept, so we return a single page (page 1).
    Page number will be set to None on chunks to signal "no page data".

    Args:
        file_path: Path to the .txt file.

    Returns:
        ExtractionResult with a single (1, normalized_text) entry.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as UTF-8.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    logger.info(f"Starting TXT extraction | file={file_path.name}")

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"Failed to read text file '{file_path.name}': {e}") from e

    cleaned = normalize_text(raw_text)

    if not cleaned:
        raise ValueError(f"File '{file_path.name}' is empty after normalization.")

    logger.info(
        f"TXT extraction complete | file={file_path.name} | chars={len(cleaned)}"
    )

    return ExtractionResult(
        pages=[(1, cleaned)],
        total_characters=len(cleaned),
        total_pages=1,
        document_type="txt",
    )


# ─────────────────────────────────────────────
# Page Map Utility
# ─────────────────────────────────────────────

def _build_page_map(pages: List[Tuple[int, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Join all pages into a single string and build a character-position → page
    lookup structure.

    Returns:
        full_text: Concatenated, page-separated document text.
        page_map: Sorted list of (cumulative_start_char, page_number).
                  Used with bisect to find which page a character offset is on.

    Why concatenate pages?
      Chunking page-by-page creates isolated mini-documents. A sentence that
      spans the bottom of page 4 and the top of page 5 would be split,
      losing its meaning. Concatenation + overlap fixes this.
    """
    parts: List[str] = []
    page_map: List[Tuple[int, int]] = []
    cursor = 0

    for page_num, text in pages:
        page_map.append((cursor, page_num))
        parts.append(text)
        cursor += len(text) + 2  # +2 for the "\n\n" separator

    full_text = "\n\n".join(p for _, p in pages)
    return full_text, page_map


def _get_page_for_position(position: int, page_map: List[Tuple[int, int]]) -> Optional[int]:
    """
    Given a character position in the full concatenated text, return the
    corresponding page number using binary search on the page_map.

    Returns None if page_map is empty (e.g., txt files with no page data).
    """
    if not page_map:
        return None

    # Find the last page start that is <= position
    starts = [start for start, _ in page_map]
    idx = bisect.bisect_right(starts, position) - 1
    idx = max(0, idx)
    return page_map[idx][1]


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_document(
    extraction: ExtractionResult,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Apply sliding window chunking to an ExtractionResult.

    Algorithm:
      1. Join all pages into a single string (preserving paragraph separators).
      2. Build a page_map for O(log n) page attribution per chunk.
      3. Slide a window of `chunk_size` characters across the full text,
         advancing by (chunk_size - chunk_overlap) each step.
      4. Each chunk gets: deterministic ID, page attribution, metadata.

    Args:
        extraction: Output from extract_text_from_pdf or extract_text_from_txt.
        filename: Original filename (used in chunk IDs and metadata).
        chunk_size: Target character count per chunk. Default: 500.
        chunk_overlap: Characters shared between adjacent chunks. Default: 50.

    Returns:
        Ordered list of DocumentChunk objects.

    Engineering notes:
      - Chunk IDs are deterministic: "{file_stem}_chunk_{index:04d}"
      - Timestamps use UTC to avoid timezone ambiguity in production.
      - If the document is shorter than chunk_size, it becomes a single chunk.
      - Overlap must be < chunk_size (enforced with assertion).
    """
    assert chunk_overlap < chunk_size, (
        f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
    )

    # Step 1: Build full text and page map
    full_text, page_map = _build_page_map(extraction.pages)

    if not full_text.strip():
        logger.warning(f"Empty document after building page map | file={filename}")
        return []

    # Step 2: Compute file stem for deterministic IDs
    file_stem = Path(filename).stem
    # Sanitize: lowercase, replace spaces/dots with underscores
    file_stem = re.sub(r"[^a-z0-9_]", "_", file_stem.lower()).strip("_")

    # Step 3: Recursive Character Splitting
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    chunks: List[DocumentChunk] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    
    chunk_texts = splitter.split_text(full_text)
    
    current_search_start = 0

    for chunk_index, chunk_text in enumerate(chunk_texts):
        clean_text = chunk_text.strip()

        # Skip empty or extremely tiny chunks
        if len(clean_text) < 40:
            continue

        # Find position to attribute page correctly
        pos = full_text.find(chunk_text, current_search_start)
        if pos == -1:
            pos = current_search_start

        page_num = _get_page_for_position(pos, page_map)

        # TXT files have no page numbers
        if extraction.document_type == "txt":
            page_num = None

        chunks.append(
            DocumentChunk(
                chunk_id=f"{file_stem}_chunk_{chunk_index:04d}",
                text=clean_text,
                filename=filename,
                document_type=extraction.document_type,
                page_number=page_num,
                chunk_index=chunk_index,
                char_count=len(clean_text),
                timestamp=ingestion_timestamp,
            )
        )

        # Advance search start, accounting for overlap
        current_search_start = max(0, pos + len(chunk_text) - chunk_overlap - 100)
        
    avg_chunk_size = (
        round(
            sum(chunk.char_count for chunk in chunks) / len(chunks),
            1
        )
        if chunks else 0
    )

    logger.info(
        f"Chunking complete | "
        f"file={filename} | "
        f"total_chars={len(full_text)} | "
        f"chunks={len(chunks)} | "
        f"avg_chunk_size={avg_chunk_size} | "
        f"chunk_size={chunk_size} | "
        f"overlap={chunk_overlap}"
    )


    return chunks
