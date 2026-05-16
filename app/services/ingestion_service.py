"""
app/services/ingestion_service.py
===================================
Complete ingestion orchestration service.

Architectural role:
  - This service is the ONLY upload orchestration layer.
  - It coordinates:
        extraction
        → chunking
        → embedding
        → vector storage
  - FastAPI routes must ONLY call this service.
  - This preserves thin-route architecture.

This service does NOT:
  - perform retrieval
  - call Groq
  - build prompts
  - know about FastAPI request objects
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.document_processor import (
    DocumentChunk,
    ExtractionResult,
    chunk_document,
    extract_text_from_pdf,
    extract_text_from_txt,
)
from embeddings.embedder import EmbeddingEngine, get_embedding_engine
from vectorstore.chroma_store import AddResult, VectorStore

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
}

# Global write lock
# ChromaDB PersistentClient is not safe for concurrent writes.
_vectorstore_write_lock = threading.Lock()


# ─────────────────────────────────────────────
# Result Structures
# ─────────────────────────────────────────────

@dataclass
class DocumentStatistics:
    total_pages: int
    total_chunks: int
    total_characters: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int


@dataclass
class IngestionResult:
    success: bool
    filename: str
    document_type: str
    chunks: List[DocumentChunk]
    stats: DocumentStatistics
    processing_time_ms: float

    embeddings_created: int = 0
    vectors_stored: int = 0

    error: Optional[str] = None


# ─────────────────────────────────────────────
# Ingestion Service
# ─────────────────────────────────────────────

class IngestionService:
    """
    Full ingestion orchestrator.

    Pipeline:
        validate
        → extract
        → chunk
        → embed
        → store
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedder: Optional[EmbeddingEngine] = None,
        vectorstore: Optional[VectorStore] = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Dependency injection
        self.embedder = embedder or get_embedding_engine()
        self.vectorstore = vectorstore or VectorStore()

        logger.debug(
            f"IngestionService initialized | "
            f"chunk_size={self.chunk_size} | "
            f"overlap={self.chunk_overlap}"
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def ingest_file(
        self,
        file_path: str | Path,
        filename: str,
    ) -> IngestionResult:
        """
        Complete upload pipeline.

        Stages:
            1. Validate
            2. Extract
            3. Chunk
            4. Embed
            5. Store
        """

        start_time = time.perf_counter()
        file_path = Path(file_path)

        logger.info(
            f"Ingestion started | file={filename}",
            extra={"ai_pipeline": True},
        )

        # ── Stage 1: Validation ──────────────────

        validation_error = self._validate_file(file_path, filename)
        if validation_error:
            return self._failure_result(
                filename=filename,
                error_message=validation_error,
                start_time=start_time,
            )

        document_type = SUPPORTED_EXTENSIONS[file_path.suffix.lower()]

        # ── Stage 2: Extraction ──────────────────

        try:
            extraction = self._extract_text(file_path, document_type)
        except Exception as e:
            logger.error(
                f"Extraction failed | file={filename} | error={e}",
                extra={"ai_pipeline": True},
            )

            return self._failure_result(
                filename=filename,
                error_message=f"Extraction error: {e}",
                start_time=start_time,
            )

        # ── Stage 3: Chunking ────────────────────

        try:
            chunks = chunk_document(
                extraction=extraction,
                filename=filename,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except Exception as e:
            logger.error(
                f"Chunking failed | file={filename} | error={e}",
                extra={"ai_pipeline": True},
            )

            return self._failure_result(
                filename=filename,
                error_message=f"Chunking error: {e}",
                start_time=start_time,
            )

        if not chunks:
            return self._failure_result(
                filename=filename,
                error_message="No chunks were generated from the document.",
                start_time=start_time,
            )

        # ── Stage 4: Embedding ───────────────────

        try:
            embeddings = self.embedder.encode_chunks(chunks)
        except Exception as e:
            logger.error(
                f"Embedding failed | file={filename} | error={e}",
                extra={"ai_pipeline": True},
            )

            return self._failure_result(
                filename=filename,
                error_message=f"Embedding error: {e}",
                start_time=start_time,
            )

        # ── Stage 5: Vector Storage ──────────────

        try:
            # ChromaDB writes must be serialized
            with _vectorstore_write_lock:
                add_result: AddResult = self.vectorstore.add_documents(
                    chunks=chunks,
                    embeddings=embeddings,
                )
        except Exception as e:
            logger.error(
                f"Vector storage failed | file={filename} | error={e}",
                extra={"ai_pipeline": True},
            )

            return self._failure_result(
                filename=filename,
                error_message=f"Vector storage error: {e}",
                start_time=start_time,
            )

        if not add_result.success:
            return self._failure_result(
                filename=filename,
                error_message=add_result.error or "Unknown vectorstore error.",
                start_time=start_time,
            )

        # ── Statistics ───────────────────────────

        stats = self._compute_statistics(extraction, chunks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Ingestion complete | "
            f"file={filename} | "
            f"chunks={stats.total_chunks} | "
            f"vectors={add_result.chunks_added} | "
            f"time={elapsed_ms:.1f}ms",
            extra={"ai_pipeline": True},
        )

        return IngestionResult(
            success=True,
            filename=filename,
            document_type=document_type,
            chunks=chunks,
            stats=stats,
            processing_time_ms=elapsed_ms,
            embeddings_created=len(embeddings),
            vectors_stored=add_result.chunks_added,
        )

    # ─────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────

    def _validate_file(
        self,
        file_path: Path,
        filename: str,
    ) -> Optional[str]:

        if not file_path.exists():
            return f"File not found: {filename}"

        if not file_path.is_file():
            return f"Path is not a file: {filename}"

        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return (
                f"Unsupported file type '{ext}'. "
                f"Supported formats: {supported}"
            )

        if file_path.stat().st_size == 0:
            return f"File '{filename}' is empty."

        return None

    def _extract_text(
        self,
        file_path: Path,
        document_type: str,
    ) -> ExtractionResult:

        if document_type == "pdf":
            return extract_text_from_pdf(file_path)

        elif document_type == "txt":
            return extract_text_from_txt(file_path)

        raise ValueError(f"Unknown document type: {document_type}")

    def _compute_statistics(
        self,
        extraction: ExtractionResult,
        chunks: List[DocumentChunk],
    ) -> DocumentStatistics:

        char_counts = [chunk.char_count for chunk in chunks]

        return DocumentStatistics(
            total_pages=extraction.total_pages,
            total_chunks=len(chunks),
            total_characters=extraction.total_characters,
            avg_chunk_size=round(sum(char_counts) / len(char_counts), 1),
            min_chunk_size=min(char_counts),
            max_chunk_size=max(char_counts),
        )

    def _failure_result(
        self,
        filename: str,
        error_message: str,
        start_time: float,
    ) -> IngestionResult:

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            f"Ingestion failed | "
            f"file={filename} | "
            f"error={error_message} | "
            f"time={elapsed_ms:.1f}ms",
            extra={"ai_pipeline": True},
        )

        return IngestionResult(
            success=False,
            filename=filename,
            document_type="unknown",
            chunks=[],
            stats=DocumentStatistics(
                total_pages=0,
                total_chunks=0,
                total_characters=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
            ),
            processing_time_ms=elapsed_ms,
            embeddings_created=0,
            vectors_stored=0,
            error=error_message,
        )