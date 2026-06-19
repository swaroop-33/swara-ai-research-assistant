import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import chromadb

from app.core.config import settings
from app.core.logging import get_logger
from app.document_processor import DocumentChunk
from app.schemas.response import RetrievedChunk
logger = get_logger(__name__)

_NO_PAGE_SENTINEL = -1


@dataclass
class AddResult:
    success: bool
    chunks_added: int
    collection_name: str
    insertion_time_ms: float
    error: Optional[str] = None


@dataclass
class CollectionStats:
    collection_name: str
    total_chunks: int
    persist_directory: str
    is_ready: bool


class VectorStore:

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):

        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = (
            collection_name or settings.chroma_collection_name
        )

        self._client: Optional[chromadb.PersistentClient] = None

        self._initialize()

    # ─────────────────────────────────────

    def _initialize(self) -> None:

        persist_path = Path(self._persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initializing ChromaDB | "
            f"dir={self._persist_dir} | "
            f"base_collection={self._collection_name}"
        )

        self._client = chromadb.PersistentClient(
            path=str(persist_path)
        )

        logger.info("ChromaDB persistent client ready")

    # ─────────────────────────────────────

    def _get_collection(self, session_id: str):
        """
        Dynamically fetches or creates an isolated collection for the given session.
        If session_id is missing or 'default', returns the legacy base collection.
        """
        if not session_id or session_id == "default":
            name = self._collection_name
        else:
            # ChromaDB requires collections to be alphanumeric and underscores/hyphens
            # Max length is 63 characters.
            safe_id = "".join(c for c in session_id if c.isalnum() or c in ('_', '-'))
            name = f"{self._collection_name}_{safe_id}"[:63]

        collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection, name

    # ─────────────────────────────────────

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        session_id: str = "default",
    ) -> AddResult:

        collection, c_name = self._get_collection(session_id)

        if len(chunks) != len(embeddings):

            return AddResult(
                success=False,
                chunks_added=0,
                collection_name=c_name,
                insertion_time_ms=0.0,
                error="Chunks/embeddings mismatch",
            )

        if not chunks:

            return AddResult(
                success=False,
                chunks_added=0,
                collection_name=c_name,
                insertion_time_ms=0.0,
                error="Empty chunk list",
            )

        start = time.perf_counter()

        try:

            logger.info(
                f"Vector insert started | "
                f"collection={c_name} | "
                f"chunks={len(chunks)} | "
                f"embeddings={len(embeddings)} | "
                f"first_chunk_id={chunks[0].chunk_id}"
            )
            
            collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[
                    self._chunk_to_metadata(c)
                    for c in chunks
                ],
            )

        except Exception as e:

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            logger.error(
                f"Vector storage failed | collection={c_name} | error={e}"
            )

            return AddResult(
                success=False,
                chunks_added=0,
                collection_name=c_name,
                insertion_time_ms=elapsed,
                error=str(e),
            )
        
        collection_count = collection.count()

        logger.info(
            f"Vector insert complete | "
            f"collection={c_name} | "
            f"total_chunks={collection_count}"
        )
        
        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return AddResult(
            success=True,
            chunks_added=len(chunks),
            collection_name=c_name,
            insertion_time_ms=elapsed,
        )

    # ─────────────────────────────────────
    def reset(self, session_id: str = "default") -> bool:
        """
        Alias for clear_collection().
        """
        return self.clear_collection(session_id)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        session_id: str = "default",
    ) -> List[RetrievedChunk]:

        collection, c_name = self._get_collection(session_id)

        try:
            collection_size = collection.count()

        except Exception:

            logger.warning(
                f"Collection {c_name} missing during query."
            )

            return []

        if collection_size == 0:
            return []

        effective_n = min(
            n_results,
            collection_size,
        )

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results: List[RetrievedChunk] = []

        for idx, (
            chunk_id,
            text,
            meta,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ):

            page_num = meta.get(
                "page_number",
                _NO_PAGE_SENTINEL,
            )

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    filename=meta.get(
                        "filename",
                        "unknown",
                    ),
                    page_number=(
                        None
                        if page_num == _NO_PAGE_SENTINEL
                        else int(page_num)
                    ),
                    document_type=meta.get(
                        "document_type",
                        "unknown",
                    ),
                    similarity_score=round(
                        self._distance_to_similarity(
                            distance
                        ),
                        6,
                    ),
                    rank=idx + 1,
                    timestamp=meta.get(
                        "timestamp",
                        "",
                    ),
                )
            )

        return results

    # ─────────────────────────────────────

    def get_stats(self, session_id: str = "default") -> CollectionStats:

        collection, c_name = self._get_collection(session_id)

        try:
            count = collection.count()
            ready = True

        except Exception:
            count = 0
            ready = False

        return CollectionStats(
            collection_name=c_name,
            total_chunks=count,
            persist_directory=str(
                Path(self._persist_dir).resolve()
            ),
            is_ready=ready,
        )

    # ─────────────────────────────────────

    def clear_collection(self, session_id: str = "default") -> bool:

        collection, c_name = self._get_collection(session_id)

        try:

            self._client.delete_collection(
                name=c_name
            )

            # Recreate an empty one
            self._client.get_or_create_collection(
                name=c_name,
                metadata={
                    "hnsw:space": "cosine"
                },
            )

            logger.info(
                f"Collection reset | "
                f"name={c_name}"
            )

            return True

        except Exception as e:

            logger.error(
                f"Collection reset failed | name={c_name} | error={e}"
            )

            raise RuntimeError(
                f"Collection clear failed: {e}"
            ) from e

    # ─────────────────────────────────────

    def _chunk_to_metadata(
        self,
        chunk: DocumentChunk,
    ) -> dict:

        return {
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
            "page_number": (
                chunk.page_number
                if chunk.page_number is not None
                else _NO_PAGE_SENTINEL
            ),
            "timestamp": chunk.timestamp,
            "document_type": chunk.document_type,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
        }

    # ─────────────────────────────────────

    @staticmethod
    def _distance_to_similarity(
        distance: float,
    ) -> float:

        return max(
            0.0,
            min(1.0, 1.0 - distance),
        )


# ─────────────────────────────────────────────
# Singleton Factory
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """
    Return the application-wide VectorStore singleton.

    CRITICAL: All services (IngestionService, RetrievalService, upload route)
    MUST use this function instead of calling VectorStore() directly.

    Reason: chromadb.PersistentClient maintains an in-process HNSW index.
    Multiple PersistentClient instances pointing to the same directory in the
    same process do NOT share this index — writes by one instance are invisible
    to another until the process restarts (and ChromaDB reloads from disk).

    A single shared instance eliminates this fragmentation entirely.
    """
    logger.info("Creating VectorStore singleton...")
    return VectorStore()
