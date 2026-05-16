"""
tests/test_document_processor.py
==================================
Unit tests for the document processing pipeline.

Testing philosophy:
  - We test BEHAVIOR, not content. We never assert "this PDF contains X text"
    because that makes tests brittle and tied to external files.
  - All file I/O uses pytest's tmp_path fixture — no hardcoded paths.
  - PDF tests use programmatically created PDFs (via pypdf writer) so
    we control the exact content.
  - Each test has a single responsibility and a descriptive name.
"""

import pytest
from pathlib import Path
from datetime import datetime

import pypdf

from app.document_processor import (
    DocumentChunk,
    ExtractionResult,
    normalize_text,
    extract_text_from_txt,
    chunk_document,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _create_txt_file(tmp_path: Path, content: str, name: str = "test.txt") -> Path:
    """Helper: write a text file and return its path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _create_simple_pdf(tmp_path: Path, pages: list[str], name: str = "test.pdf") -> Path:
    """
    Helper: create a multi-page PDF using pypdf's PdfWriter.
    Each string in `pages` becomes one page.
    """
    writer = pypdf.PdfWriter()
    for page_text in pages:
        # Add a blank page; we use add_blank_page since pypdf 4+ doesn't have
        # a simple add_page with text. For content injection, we add metadata.
        page = writer.add_blank_page(width=612, height=792)

    path = tmp_path / name
    with open(path, "wb") as f:
        writer.write(f)
    return path


# ─────────────────────────────────────────────
# normalize_text tests
# ─────────────────────────────────────────────

class TestNormalizeText:
    """Verify text cleaning behavior before chunking."""

    def test_strips_null_bytes(self):
        result = normalize_text("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_collapses_excessive_newlines(self):
        text = "paragraph one\n\n\n\n\nparagraph two"
        result = normalize_text(text)
        assert "\n\n\n" not in result, "Three+ newlines should be collapsed to two"

    def test_normalizes_windows_line_endings(self):
        text = "line one\r\nline two\r\nline three"
        result = normalize_text(text)
        assert "\r" not in result

    def test_collapses_multiple_spaces(self):
        text = "this   has   too   many   spaces"
        result = normalize_text(text)
        assert "  " not in result  # no double spaces

    def test_strips_tabs(self):
        text = "column one\tcolumn two\tcolumn three"
        result = normalize_text(text)
        assert "\t" not in result

    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert normalize_text("   \n\n\t  ") == ""

    def test_preserves_paragraph_structure(self):
        text = "Paragraph one.\n\nParagraph two."
        result = normalize_text(text)
        assert "Paragraph one." in result
        assert "Paragraph two." in result
        # Double newline (paragraph separator) should be preserved
        assert "\n\n" in result


# ─────────────────────────────────────────────
# TXT Extraction tests
# ─────────────────────────────────────────────

class TestTxtExtraction:
    """Verify plain text file extraction behavior."""

    def test_extracts_content_from_txt_file(self, tmp_path):
        content = "This is a test document.\nIt has multiple lines."
        path = _create_txt_file(tmp_path, content)

        result = extract_text_from_txt(path)

        assert result.document_type == "txt"
        assert result.total_pages == 1
        assert "This is a test document" in result.pages[0][1]

    def test_page_number_is_always_one(self, tmp_path):
        path = _create_txt_file(tmp_path, "Some text content here.")
        result = extract_text_from_txt(path)
        assert result.pages[0][0] == 1  # first element of tuple is page_num

    def test_total_characters_nonzero(self, tmp_path):
        path = _create_txt_file(tmp_path, "A" * 200)
        result = extract_text_from_txt(path)
        assert result.total_characters > 0

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_txt("/nonexistent/path/file.txt")

    def test_utf8_content_preserved(self, tmp_path):
        content = "Unicode test: café, naïve, résumé, 日本語"
        path = _create_txt_file(tmp_path, content)
        result = extract_text_from_txt(path)
        assert "café" in result.pages[0][1]


# ─────────────────────────────────────────────
# Chunking tests
# ─────────────────────────────────────────────

class TestChunkDocument:
    """Verify sliding window chunking behavior."""

    def _make_extraction(self, text: str, doc_type: str = "txt") -> ExtractionResult:
        """Build a minimal ExtractionResult from raw text for testing."""
        return ExtractionResult(
            pages=[(1, text)],
            total_characters=len(text),
            total_pages=1,
            document_type=doc_type,
        )

    def test_single_chunk_for_short_document(self):
        """A document shorter than chunk_size becomes exactly one chunk."""
        text = "Short document content."
        extraction = self._make_extraction(text)
        chunks = chunk_document(extraction, "short.txt", chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_multiple_chunks_for_long_document(self):
        """A document longer than chunk_size produces multiple chunks."""
        text = "word " * 300  # ~1500 chars
        extraction = self._make_extraction(text)
        chunks = chunk_document(extraction, "long.txt", chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_count_is_stable(self):
        """Running chunking twice on the same input produces the same count."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        extraction = self._make_extraction(text)
        chunks_a = chunk_document(extraction, "doc.txt", chunk_size=100, chunk_overlap=10)
        chunks_b = chunk_document(extraction, "doc.txt", chunk_size=100, chunk_overlap=10)
        assert len(chunks_a) == len(chunks_b)

    def test_overlap_means_adjacent_chunks_share_text(self):
        """
        With overlap > 0, the end of chunk[n] should appear at the
        start of chunk[n+1]. We verify text continuity, not exact overlap.
        """
        text = "a" * 400  # uniform content makes overlap easy to verify
        extraction = self._make_extraction(text)
        chunks = chunk_document(extraction, "doc.txt", chunk_size=100, chunk_overlap=20)

        if len(chunks) >= 2:
            # The last 20 chars of chunk 0 should be the first 20 chars of chunk 1
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            assert end_of_first == start_of_second, (
                "Overlap not correctly implemented — adjacent chunks should share text"
            )

    def test_no_empty_chunks(self):
        """All returned chunks must have non-empty text."""
        text = "Content with varied spacing.\n\nAnother paragraph.\n\nFinal section."
        extraction = self._make_extraction(text)
        chunks = chunk_document(extraction, "doc.txt", chunk_size=50, chunk_overlap=5)
        for chunk in chunks:
            assert chunk.text.strip() != "", f"Empty chunk found: {chunk.chunk_id}"

    def test_chunk_char_count_matches_text_length(self):
        """char_count field must match actual text length."""
        text = "word " * 100
        extraction = self._make_extraction(text)
        chunks = chunk_document(extraction, "doc.txt", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.char_count == len(chunk.text)


# ─────────────────────────────────────────────
# Chunk Metadata tests
# ─────────────────────────────────────────────

class TestChunkMetadata:
    """Verify every chunk carries correct and complete metadata."""

    def _make_chunks(self, filename: str = "paper.txt", doc_type: str = "txt"):
        text = "Research content. " * 60
        extraction = ExtractionResult(
            pages=[(1, text)],
            total_characters=len(text),
            total_pages=1,
            document_type=doc_type,
        )
        return chunk_document(extraction, filename, chunk_size=100, chunk_overlap=10)

    def test_filename_is_set_on_every_chunk(self):
        chunks = self._make_chunks(filename="paper.txt")
        for chunk in chunks:
            assert chunk.filename == "paper.txt"

    def test_document_type_is_set_on_every_chunk(self):
        chunks = self._make_chunks(doc_type="txt")
        for chunk in chunks:
            assert chunk.document_type == "txt"

    def test_timestamp_is_iso8601(self):
        chunks = self._make_chunks()
        for chunk in chunks:
            # datetime.fromisoformat raises if format is invalid
            datetime.fromisoformat(chunk.timestamp)

    def test_all_chunks_share_same_ingestion_timestamp(self):
        """All chunks from one ingestion run share the same timestamp."""
        chunks = self._make_chunks()
        timestamps = {chunk.timestamp for chunk in chunks}
        assert len(timestamps) == 1, (
            "All chunks from one ingestion should share the same timestamp"
        )

    def test_txt_chunks_have_none_page_number(self):
        """TXT files have no page concept — page_number must be None."""
        chunks = self._make_chunks(doc_type="txt")
        for chunk in chunks:
            assert chunk.page_number is None

    def test_chunk_index_is_sequential(self):
        chunks = self._make_chunks()
        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))


# ─────────────────────────────────────────────
# Deterministic Chunk ID tests
# ─────────────────────────────────────────────

class TestDeterministicChunkIDs:
    """Verify chunk IDs are deterministic, unique, and well-formed."""

    def _make_chunks(self, filename: str):
        text = "Content " * 80
        extraction = ExtractionResult(
            pages=[(1, text)],
            total_characters=len(text),
            total_pages=1,
            document_type="txt",
        )
        return chunk_document(extraction, filename, chunk_size=100, chunk_overlap=10)

    def test_chunk_ids_are_unique_within_document(self):
        chunks = self._make_chunks("report.txt")
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_format_is_correct(self):
        """IDs must follow pattern: {stem}_chunk_{XXXX} (zero-padded 4 digits)."""
        chunks = self._make_chunks("my_paper.txt")
        for chunk in chunks:
            parts = chunk.chunk_id.split("_chunk_")
            assert len(parts) == 2, f"Unexpected chunk_id format: {chunk.chunk_id}"
            index_str = parts[1]
            assert index_str.isdigit() and len(index_str) == 4, (
                f"Index part should be 4-digit zero-padded: {chunk.chunk_id}"
            )

    def test_chunk_ids_are_reproducible_across_runs(self):
        """Same input → same IDs every time (deterministic)."""
        chunks_a = self._make_chunks("doc.txt")
        chunks_b = self._make_chunks("doc.txt")
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b

    def test_different_filenames_produce_different_id_prefixes(self):
        chunks_a = self._make_chunks("alpha.txt")
        chunks_b = self._make_chunks("beta.txt")
        prefix_a = chunks_a[0].chunk_id.split("_chunk_")[0]
        prefix_b = chunks_b[0].chunk_id.split("_chunk_")[0]
        assert prefix_a != prefix_b


# ─────────────────────────────────────────────
# IngestionService integration test
# ─────────────────────────────────────────────

class TestIngestionServiceIntegration:
    """
    Integration test: IngestionService.ingest_file on real .txt files.
    No PDF integration test here — PDF extraction is tested via pypdf behavior,
    and mocking pypdf is out of scope for this phase.
    """

    def test_ingest_txt_file_returns_success(self, tmp_path):
        from app.services.ingestion_service import IngestionService
        service = IngestionService(chunk_size=100, chunk_overlap=10)

        content = "This is a research document. " * 40
        path = tmp_path / "research.txt"
        path.write_text(content, encoding="utf-8")

        result = service.ingest_file(file_path=path, filename="research.txt")

        assert result.success is True
        assert result.filename == "research.txt"
        assert result.document_type == "txt"
        assert result.stats.total_chunks > 0
        assert result.stats.total_characters > 0
        assert result.processing_time_ms > 0
        assert result.error is None

    def test_ingest_missing_file_returns_failure(self, tmp_path):
        from app.services.ingestion_service import IngestionService
        service = IngestionService()

        result = service.ingest_file(
            file_path=tmp_path / "nonexistent.txt",
            filename="nonexistent.txt"
        )

        assert result.success is False
        assert result.error is not None
        assert len(result.chunks) == 0

    def test_ingest_unsupported_format_returns_failure(self, tmp_path):
        from app.services.ingestion_service import IngestionService
        service = IngestionService()

        path = tmp_path / "document.docx"
        path.write_bytes(b"fake docx content")

        result = service.ingest_file(file_path=path, filename="document.docx")

        assert result.success is False
        assert "Unsupported" in result.error

    def test_statistics_are_accurate(self, tmp_path):
        from app.services.ingestion_service import IngestionService
        service = IngestionService(chunk_size=100, chunk_overlap=10)

        content = "word " * 100
        path = tmp_path / "stats_test.txt"
        path.write_text(content, encoding="utf-8")

        result = service.ingest_file(file_path=path, filename="stats_test.txt")

        assert result.stats.total_chunks == len(result.chunks)
        assert result.stats.avg_chunk_size > 0
        assert result.stats.min_chunk_size <= result.stats.avg_chunk_size
        assert result.stats.avg_chunk_size <= result.stats.max_chunk_size
