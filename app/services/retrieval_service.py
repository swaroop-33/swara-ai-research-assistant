"""
app/services/retrieval_service.py
===================================
Semantic retrieval service — orchestrates query embedding + vector search.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.response import RetrievedChunk
from embeddings.embedder import (
    EmbeddingEngine,
    get_embedding_engine,
)
from vectorstore.chroma_store import VectorStore

logger = get_logger(__name__)

# =========================================================
# RETRIEVAL CONFIG
# =========================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.22

HIGH_PRECISION_THRESHOLD = 0.30

MAX_INTERNAL_RETRIEVAL = 12

# =========================================================
# RESULT STRUCTURE
# =========================================================

@dataclass
class RetrievalResult:

    query: str
    chunks: List[RetrievedChunk]
    retrieval_time_ms: float
    embedding_time_ms: float
    search_time_ms: float
    total_chunks_in_store: int
    is_empty: bool


# =========================================================
# RETRIEVAL SERVICE
# =========================================================

class RetrievalService:

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
    ):

        self._embedding_engine = embedding_engine
        self._vector_store = vector_store

        logger.info(
            "RetrievalService initialized"
        )

    # =====================================================
    # MAIN RETRIEVAL PIPELINE
    # =====================================================

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> RetrievalResult:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        k = n_results or settings.top_k_results

        total_start = time.perf_counter()

        logger.info(
            f"Retrieval started | "
            f"query_len={len(query)} | "
            f"k={k}",
            extra={"ai_pipeline": True},
        )

        # =================================================
        # STEP 1 — STORE VALIDATION
        # =================================================

        stats = self._vector_store.get_stats()

        if stats.total_chunks == 0:

            logger.warning(
                "Retrieval attempted on empty vector store.",
                extra={"ai_pipeline": True},
            )

            return RetrievalResult(
                query=query,
                chunks=[],
                retrieval_time_ms=0.0,
                embedding_time_ms=0.0,
                search_time_ms=0.0,
                total_chunks_in_store=0,
                is_empty=True,
            )

        # =================================================
        # STEP 2 — QUERY EMBEDDING
        # =================================================

        embed_start = time.perf_counter()

        query_vector = (
            self._embedding_engine.encode_query(
                query
            )
        )

        embedding_time_ms = (
            time.perf_counter() - embed_start
        ) * 1000

        # =================================================
        # STEP 3 — VECTOR SEARCH
        # =================================================

        search_start = time.perf_counter()

        internal_k = min(
            k * 2,
            MAX_INTERNAL_RETRIEVAL,
        )

        chunks = self._vector_store.query(
            query_vector,
            n_results=internal_k,
        )

        # =================================================
        # STEP 4 — DYNAMIC THRESHOLDING
        # =================================================

        query_lower = query.lower()

        factual_keywords = [
            "what",
            "when",
            "where",
            "who",
            "population",
            "date",
            "definition",
        ]

        is_factual_query = any(
            keyword in query_lower
            for keyword in factual_keywords
        )

        similarity_threshold = (
            HIGH_PRECISION_THRESHOLD
            if is_factual_query
            else DEFAULT_SIMILARITY_THRESHOLD
        )

        filtered_chunks = [
            chunk
            for chunk in chunks
            if chunk.similarity_score
            >= similarity_threshold
        ]

        removed_chunks = (
            len(chunks)
            - len(filtered_chunks)
        )

        if removed_chunks > 0:

            logger.info(
                f"Low-confidence chunks filtered | "
                f"removed={removed_chunks} | "
                f"remaining={len(filtered_chunks)} | "
                f"threshold={similarity_threshold}",
                extra={"ai_pipeline": True},
            )

        # =================================================
        # STEP 5 — FINAL RANKING
        # =================================================

        filtered_chunks = sorted(
            filtered_chunks,
            key=lambda x: (
                x.similarity_score,
                len(x.text),
            ),
            reverse=True,
        )

        chunks = filtered_chunks[:k]

        search_time_ms = (
            time.perf_counter() - search_start
        ) * 1000

        total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000

        logger.info(
            f"Retrieval complete | "
            f"chunks={len(chunks)} | "
            f"embed={embedding_time_ms:.1f}ms | "
            f"search={search_time_ms:.1f}ms | "
            f"total={total_time_ms:.1f}ms | "
            f"top_score={chunks[0].similarity_score if chunks else 'N/A'}",
            extra={"ai_pipeline": True},
        )

        if not chunks:

            logger.warning(
                "No sufficiently relevant chunks retrieved."
            )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
            total_chunks_in_store=stats.total_chunks,
            is_empty=len(chunks) == 0,
        )

    # =====================================================
    # VECTOR STORE STATS
    # =====================================================

    def get_store_stats(self):

        return self._vector_store.get_stats()

    # =====================================================
    # CLEAR VECTOR STORE
    # =====================================================

    def clear_store(self) -> None:

        logger.info(
            "Clearing vector store via RetrievalService"
        )

        self._vector_store.clear_collection()


# =========================================================
# SINGLETON FACTORY
# =========================================================

@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:

    logger.info(
        "Creating RetrievalService singleton..."
    )

    return RetrievalService(
        embedding_engine=get_embedding_engine(),
        vector_store=VectorStore(),
    )