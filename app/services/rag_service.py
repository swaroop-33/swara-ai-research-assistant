"""
app/services/rag_service.py
============================
RAG orchestration service — coordinates retrieval + LLM answer generation.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from app.core.logging import get_logger
from app.schemas.response import RetrievedChunk
from app.services.retrieval_service import (
    RetrievalResult,
    RetrievalService,
)

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 7000


# ─────────────────────────────────────────────
# Response Structure
# ─────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    question: str
    retrieved_chunks: List[RetrievedChunk]
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    fallback_used: bool = False


# ─────────────────────────────────────────────
# RAG Service
# ─────────────────────────────────────────────

class RAGService:
    """
    Full RAG orchestration layer.

    Flow:
        Question
        → Retrieval
        → Context Build
        → LLM Generation
        → Structured Response
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service=None,
    ):
        self._retrieval = retrieval_service
        self._llm = llm_service

        logger.info(
            f"RAGService initialized | "
            f"llm_ready={llm_service is not None}"
        )

    # ─────────────────────────────────────────

    def answer(
        self,
        question: str,
        n_results: Optional[int] = None,
        chat_history: Optional[list] = None,
    ) -> RAGResponse:

        total_start = time.perf_counter()

        logger.info(
            f"RAG pipeline started | question_len={len(question)}",
            extra={"ai_pipeline": True},
        )

        # =================================================
        # STEP 1 — RETRIEVAL
        # =================================================

        retrieval: RetrievalResult = self._retrieval.retrieve(
            query=question,
            n_results=n_results,
        )

        # =================================================
        # NO DOCUMENTS
        # =================================================

        if retrieval.total_chunks_in_store == 0:

            return RAGResponse(
                answer=(
                    "⚠️ No document has been uploaded yet. "
                    "Please upload a PDF or text file first."
                ),
                question=question,
                retrieved_chunks=[],
                model_used="none",
                retrieval_time_ms=retrieval.retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=(
                    time.perf_counter() - total_start
                ) * 1000,
                fallback_used=False,
            )

        # =================================================
        # NO RELEVANT RETRIEVALS
        # =================================================

        if retrieval.is_empty:

            return RAGResponse(
                answer=(
                    "The uploaded documents do not contain "
                    "sufficient relevant information to answer "
                    "this question."
                ),
                question=question,
                retrieved_chunks=[],
                model_used="none",
                retrieval_time_ms=retrieval.retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=(
                    time.perf_counter() - total_start
                ) * 1000,
                fallback_used=True,
            )

        # =================================================
        # STEP 2 — CONTEXT BUILD
        # =================================================

        context = self._build_context(
            retrieval.chunks
        )

        # =================================================
        # STEP 3 — GENERATION
        # =================================================

        generation_start = time.perf_counter()

        if self._llm is None:

            answer_text = (
                "[LLM not connected]\n\n"
                + "\n\n".join(
                    chunk.text
                    for chunk in retrieval.chunks
                )
            )

            model_used = "none"
            fallback_used = False

        else:

            llm_result = self._llm.generate(
                question=question,
                context=context,
                chat_history=chat_history,
            )

            answer_text = llm_result.answer
            model_used = llm_result.model_used
            fallback_used = llm_result.fallback_used

        generation_time_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000

        logger.info(
            f"RAG pipeline complete | "
            f"chunks={len(retrieval.chunks)} | "
            f"model={model_used}",
            extra={"ai_pipeline": True},
        )

        return RAGResponse(
            answer=answer_text,
            question=question,
            retrieved_chunks=retrieval.chunks,
            model_used=model_used,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            fallback_used=fallback_used,
        )

    # ─────────────────────────────────────────
    # CONTEXT BUILDER
    # ─────────────────────────────────────────

    def _build_context(
        self,
        chunks: List[RetrievedChunk],
    ) -> str:

        # =============================================
        # SORT STRONGEST EVIDENCE FIRST
        # =============================================

        chunks = sorted(
            chunks,
            key=lambda x: x.similarity_score,
            reverse=True,
        )

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

            # =========================================
            # PRIORITY LABELING
            # =========================================

            if idx == 1:

                priority = "PRIMARY EVIDENCE"

            elif idx <= 3:

                priority = "SUPPORTING EVIDENCE"

            else:

                priority = "ADDITIONAL CONTEXT"

            # =========================================
            # CONTEXT ENTRY
            # =========================================

            entry = (
                f"=== {priority} ===\n"
                f"Source File: {chunk.filename}"
                f"{page_info}\n"
                f"Relevance Score: "
                f"{chunk.similarity_score:.2%}\n\n"
                f"{chunk.text.strip()}"
            )

            # =========================================
            # CONTEXT LIMIT
            # =========================================

            if (
                total_chars + len(entry)
                > MAX_CONTEXT_CHARS
            ):
                break

            parts.append(entry)

            total_chars += len(entry)

        # =============================================
        # SYNTHESIS INSTRUCTION
        # =============================================

        synthesis_instruction = """
Use the retrieved evidence above to produce a coherent grounded answer.

Prioritize:
- strongest evidence first
- cross-chunk synthesis
- narrative consistency
- grounded interpretation

Do not fabricate unsupported details.
"""

        return (
            synthesis_instruction
            + "\n\n"
            + "\n\n".join(parts)
        )


# ─────────────────────────────────────────────
# Singleton Factory
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_service() -> "RAGService":

    from app.services.llm_service import (
        get_llm_service,
    )

    from app.services.retrieval_service import (
        get_retrieval_service,
    )

    logger.info("Creating RAGService singleton...")

    return RAGService(
        retrieval_service=get_retrieval_service(),
        llm_service=get_llm_service(),
    )