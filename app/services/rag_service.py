# app/services/rag_service.py

"""
SWARA RAG orchestration service.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from app.core.logging import get_logger
from app.schemas.response import (
    RetrievedChunk,
    RetrievalDiagnostics,
)

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 7000


# =========================================================
# INTERNAL RESPONSE
# =========================================================

@dataclass
class RAGResponse:

    answer: str

    question: str

    rewritten_query: str

    retrieved_chunks: List[RetrievedChunk]

    model_used: str

    retrieval_time_ms: float

    generation_time_ms: float

    total_time_ms: float

    fallback_used: bool

    diagnostics: RetrievalDiagnostics


# =========================================================
# RAG SERVICE
# =========================================================

class RAGService:

    def __init__(
        self,
        retrieval_service,
        llm_service=None,
    ):

        self._retrieval = retrieval_service

        self._llm = llm_service

        logger.info(
            "RAGService initialized"
        )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def answer(
        self,
        question: str,
        n_results: Optional[int] = None,
        chat_history: Optional[list] = None,
        session_id: str = "default",
    ) -> RAGResponse:

        total_start = time.perf_counter()

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        retrieval = self._retrieval.retrieve(
            query=question,
            n_results=n_results,
            chat_history=chat_history,
            session_id=session_id,
        )

        if retrieval.total_chunks_in_store == 0:

            return self._empty_store_response(
                question,
                retrieval,
                total_start,
            )

        if retrieval.is_empty:

            return self._no_match_response(
                question,
                retrieval,
                total_start,
            )

        context = self._build_context(
            retrieval.chunks
        )

        generation_start = time.perf_counter()

        llm_result = self._llm.generate(
            question=question,
            context=context,
            chat_history=chat_history,
        )

        generation_time_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000

        similarities = [
            chunk.similarity_score
            for chunk in retrieval.chunks
        ]

        diagnostics = RetrievalDiagnostics(
            original_query=retrieval.original_query,
            expanded_query=retrieval.expanded_query,
            similarity_threshold=retrieval.similarity_threshold,
            retrieved_chunk_count=len(
                retrieval.chunks
            ),
            filtered_chunk_count=retrieval.filtered_chunk_count,
            top_similarity_score=max(similarities),
            average_similarity_score=(
                sum(similarities)
                / len(similarities)
            ),
            embedding_time_ms=retrieval.embedding_time_ms,
            search_time_ms=retrieval.search_time_ms,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            factual_query_detected=(
                retrieval.factual_query_detected
            ),
        )

        logger.info(
            f"RAG complete | "
            f"chunks={len(retrieval.chunks)} | "
            f"model={llm_result.model_used}",
            extra={"ai_pipeline": True},
        )

        return RAGResponse(
            answer=llm_result.answer,
            question=question,
            rewritten_query=retrieval.expanded_query,
            retrieved_chunks=retrieval.chunks,
            model_used=llm_result.model_used,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            fallback_used=llm_result.fallback_used,
            diagnostics=diagnostics,
        )

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================

    def _build_context(
        self,
        chunks: List[RetrievedChunk],
    ) -> str:

        parts = []

        total_chars = 0

        for idx, chunk in enumerate(
            chunks,
            start=1,
        ):

            page_info = (
                f", page {chunk.page_number}"
                if chunk.page_number
                else ""
            )

            entry = (
                f"=== EVIDENCE {idx} ===\n"
                f"Source File: {chunk.filename}"
                f"{page_info}\n"
                f"Similarity: "
                f"{chunk.similarity_score:.2%}\n\n"
                f"{chunk.text.strip()}"
            )

            if (
                total_chars + len(entry)
                > MAX_CONTEXT_CHARS
            ):
                break

            parts.append(entry)

            total_chars += len(entry)

        return "\n\n".join(parts)

    # =====================================================
    # FAILURE RESPONSES
    # =====================================================

    def _empty_store_response(
        self,
        question,
        retrieval,
        total_start,
    ):

        total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000

        diagnostics = RetrievalDiagnostics(
            original_query=question,
            expanded_query=question,
            similarity_threshold=0.0,
            retrieved_chunk_count=0,
            filtered_chunk_count=0,
            top_similarity_score=None,
            average_similarity_score=None,
            embedding_time_ms=0.0,
            search_time_ms=0.0,
            retrieval_time_ms=0.0,
            factual_query_detected=False,
        )

        return RAGResponse(
            answer=(
                "⚠️ No document has been uploaded yet."
            ),
            question=question,
            rewritten_query=question,
            retrieved_chunks=[],
            model_used="none",
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
            total_time_ms=total_time_ms,
            fallback_used=False,
            diagnostics=diagnostics,
        )

    def _no_match_response(
        self,
        question,
        retrieval,
        total_start,
    ):

        total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000

        diagnostics = RetrievalDiagnostics(
            original_query=retrieval.original_query,
            expanded_query=retrieval.expanded_query,
            similarity_threshold=retrieval.similarity_threshold,
            retrieved_chunk_count=0,
            filtered_chunk_count=retrieval.filtered_chunk_count,
            top_similarity_score=None,
            average_similarity_score=None,
            embedding_time_ms=retrieval.embedding_time_ms,
            search_time_ms=retrieval.search_time_ms,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            factual_query_detected=(
                retrieval.factual_query_detected
            ),
        )

        return RAGResponse(
            answer=(
                "The uploaded documents do not contain "
                "sufficient relevant information to answer "
                "this question."
            ),
            question=question,
            rewritten_query=retrieval.expanded_query,
            retrieved_chunks=[],
            model_used="none",
            retrieval_time_ms=retrieval.retrieval_time_ms,
            generation_time_ms=0.0,
            total_time_ms=total_time_ms,
            fallback_used=False,
            diagnostics=diagnostics,
        )


# =========================================================
# SINGLETON FACTORY
# =========================================================

@lru_cache(maxsize=1)
def get_rag_service():

    from app.services.llm_service import (
        get_llm_service,
    )

    from app.services.retrieval_service import (
        get_retrieval_service,
    )

    logger.info(
        "Creating RAGService singleton..."
    )

    return RAGService(
        retrieval_service=get_retrieval_service(),
        llm_service=get_llm_service(),
    )