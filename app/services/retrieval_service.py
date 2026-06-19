"""
app/services/retrieval_service.py
===================================
Semantic retrieval service for SWARA.
"""

import re
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

from vectorstore.chroma_store import (
    VectorStore,
    get_vector_store,
)

logger = get_logger(__name__)

# =========================================================
# RETRIEVAL CONFIG
# =========================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.18

HIGH_PRECISION_THRESHOLD = 0.24

MAX_INTERNAL_RETRIEVAL = 12


# =========================================================
# RETRIEVAL RESULT
# =========================================================

@dataclass
class RetrievalResult:

    original_query: str

    expanded_query: str

    chunks: List[RetrievedChunk]

    retrieval_time_ms: float

    embedding_time_ms: float

    search_time_ms: float

    total_chunks_in_store: int

    filtered_chunk_count: int

    similarity_threshold: float

    factual_query_detected: bool

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
        chat_history: Optional[list] = None,
        session_id: str = "default",
    ) -> RetrievalResult:

        original_query = query.strip()

        if not original_query:

            raise ValueError(
                "Query cannot be empty."
            )

        expanded_query = self._expand_query(
            query=original_query,
            chat_history=chat_history,
        )

        k = n_results or settings.top_k_results

        total_start = time.perf_counter()

        logger.info(
            f"Retrieval started | "
            f"query='{expanded_query[:120]}' | "
            f"k={k} | "
            f"session_id={session_id}",
            extra={"ai_pipeline": True},
        )

        # =================================================
        # STORE VALIDATION
        # =================================================

        stats = self._vector_store.get_stats(session_id=session_id)

        if stats.total_chunks == 0:

            logger.warning(
                "Retrieval attempted on empty vector store.",
                extra={"ai_pipeline": True},
            )

            return RetrievalResult(
                original_query=original_query,
                expanded_query=expanded_query,
                chunks=[],
                retrieval_time_ms=0.0,
                embedding_time_ms=0.0,
                search_time_ms=0.0,
                total_chunks_in_store=0,
                filtered_chunk_count=0,
                similarity_threshold=0.0,
                factual_query_detected=False,
                is_empty=True,
            )

        # =================================================
        # QUERY EMBEDDING
        # =================================================

        embed_start = time.perf_counter()

        query_vector = (
            self._embedding_engine.encode_query(
                expanded_query
            )
        )

        embedding_time_ms = (
            time.perf_counter() - embed_start
        ) * 1000

        # =================================================
        # VECTOR SEARCH
        # =================================================

        search_start = time.perf_counter()

        internal_k = min(
            k * 2,
            MAX_INTERNAL_RETRIEVAL,
        )

        chunks = self._vector_store.query(
            query_vector,
            n_results=internal_k,
            session_id=session_id,
        )

        # =================================================
        # FACTUAL DETECTION
        # =================================================

        query_lower = expanded_query.lower()

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

        # =================================================
        # DYNAMIC FILTERING
        # =================================================

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

        # =================================================
        # SMART RANKING
        # =================================================

        filtered_chunks = sorted(
            filtered_chunks,
            key=lambda x: (
                x.similarity_score,
                len(x.text),
                -x.rank,
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
            f"threshold={similarity_threshold:.2f} | "
            f"embed={embedding_time_ms:.1f}ms | "
            f"search={search_time_ms:.1f}ms",
            extra={"ai_pipeline": True},
        )

        return RetrievalResult(
            original_query=original_query,
            expanded_query=expanded_query,
            chunks=chunks,
            retrieval_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
            total_chunks_in_store=stats.total_chunks,
            filtered_chunk_count=removed_chunks,
            similarity_threshold=similarity_threshold,
            factual_query_detected=is_factual_query,
            is_empty=len(chunks) == 0,
        )

    # =====================================================
    # QUERY EXPANSION
    # =====================================================

    def _expand_query(
        self,
        query: str,
        chat_history: Optional[list] = None,
    ) -> str:

        if not chat_history:

            return query

        query_lower = query.lower()

        ambiguous_terms = [
            "he",
            "she",
            "him",
            "her",
            "they",
            "them",
            "that",
            "this",
        ]

        contains_reference = any(
            re.search(
                rf"\b{term}\b",
                query_lower,
            )
            for term in ambiguous_terms
        )

        if not contains_reference:

            return query

        recent_messages = chat_history[-6:]

        entity_pattern = r"\b[A-Z][a-zA-Z]{2,}\b"

        detected_entities = []

        for msg in reversed(recent_messages):

            # =============================================
            # SUPPORT DICTS + PYDANTIC OBJECTS
            # =============================================

            if isinstance(msg, dict):

                content = msg.get(
                    "content",
                    "",
                )

            else:

                content = getattr(
                    msg,
                    "content",
                    "",
                )

            matches = re.findall(
                entity_pattern,
                content,
            )

            for match in matches:

                if (
                    match
                    not in detected_entities
                ):

                    detected_entities.append(
                        match
                    )

        if not detected_entities:

            return query

        primary_entity = (
            detected_entities[0]
        )

        expanded_query = (
            f"{query} "
            f"(Referring to {primary_entity})"
        )

        logger.info(
            f"Query expanded | "
            f"expanded='{expanded_query}'",
            extra={"ai_pipeline": True},
        )

        return expanded_query

    # =====================================================
    # VECTOR STORE OPS
    # =====================================================

    def get_store_stats(self, session_id: str = "default"):

        return self._vector_store.get_stats(session_id=session_id)

    def clear_store(self, session_id: str = "default") -> None:

        logger.info(
            f"Clearing vector store... | session_id={session_id}"
        )

        self._vector_store.clear_collection(session_id=session_id)


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
        vector_store=get_vector_store(),
    )
