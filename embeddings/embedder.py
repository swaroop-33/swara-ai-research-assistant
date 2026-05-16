"""
embeddings/embedder.py
=======================
Semantic embedding engine using SentenceTransformers.

Architectural role:
  - This module is the ONLY place that touches SentenceTransformers.
  - It accepts text input (chunks or query strings) and returns float vectors.
  - It is completely isolated from ChromaDB, FastAPI, and Streamlit.
  - ChromaDB will receive its vectors from here in Phase 4.

Singleton design:
  - The embedding model is ~90MB and takes 1-3s to initialize.
  - Loading it on every request is unacceptable in production.
  - We use a module-level singleton with double-checked locking for
    thread-safe lazy initialization — the model loads exactly once
    per process lifetime, even under concurrent FastAPI requests.

Determinism guarantee:
  - SentenceTransformers inference is deterministic given the same model
    weights and input. No dropout is applied at inference time.
  - This means embed(text) == embed(text) always — tests can rely on this,
    and re-indexing a document produces identical vectors.

Output contract:
  - All public methods return Python List[float] or List[List[float]].
  - We explicitly convert numpy arrays from the model because:
      * numpy arrays are not JSON-serializable (API safety)
      * ChromaDB's Python client expects native Python lists
      * Pydantic schemas expect Python types for validation
"""

import threading
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.document_processor import DocumentChunk

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384          # Dimension of all-MiniLM-L6-v2 output vectors
DEFAULT_BATCH_SIZE = 32      # Optimal for CPU inference; reduce if OOM


# ── Singleton State ───────────────────────────────────────────────────────────

_instance: Optional["EmbeddingEngine"] = None
_instance_lock = threading.Lock()


# ── Engine ────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps SentenceTransformer with a production-oriented lifecycle.

    Responsibilities:
      - Load and hold the embedding model in memory (once).
      - Encode document chunks in efficient batches.
      - Encode single query strings for retrieval.
      - Validate embedding output dimensions and count.
      - Report timing metrics via structured logging.

    Do NOT instantiate directly in application code.
    Use get_embedding_engine() to always receive the singleton.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to settings.embedding_model.
            batch_size: Number of texts to encode per forward pass.
                        Larger = faster on GPU; 32 is safe for CPU.
        """
        self._model_name = model_name or settings.embedding_model
        self._batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        self._load_time_ms: float = 0.0

        self._load_model()

    # ── Model Lifecycle ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model and record load time.

        The model is downloaded from HuggingFace on first use and cached
        in ~/.cache/huggingface — subsequent loads use the local cache.
        We log the load time as an observability signal.
        """
        logger.info(
            f"Loading embedding model | model={self._model_name}",
            extra={"ai_pipeline": True},
        )

        start = time.perf_counter()
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            logger.error(
                f"Failed to load embedding model | model={self._model_name} | error={e}",
                extra={"ai_pipeline": True},
            )
            raise RuntimeError(
                f"Embedding model '{self._model_name}' could not be loaded: {e}"
            ) from e

        self._load_time_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Embedding model ready | model={self._model_name} | "
            f"dim={EMBEDDING_DIM} | load_time={self._load_time_ms:.0f}ms",
            extra={"ai_pipeline": True},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_chunks(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of document chunks.

        Processes chunks in batches of self._batch_size for memory efficiency.
        Maintains deterministic ordering: embedding[i] corresponds to chunks[i].

        Args:
            chunks: List of DocumentChunk objects from the ingestion pipeline.

        Returns:
            List of 384-dimensional float vectors, one per chunk.
            Output[i] is the embedding for chunks[i].

        Raises:
            ValueError: If chunks is empty.
            RuntimeError: If embedding generation fails or dimension mismatch.
        """
        if not chunks:
            raise ValueError("Cannot encode an empty chunk list.")

        texts = [chunk.text for chunk in chunks]
        total = len(texts)

        logger.info(
            f"Encoding chunks | count={total} | batch_size={self._batch_size}",
            extra={"ai_pipeline": True},
        )

        start = time.perf_counter()
        embeddings = self._encode_texts(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._validate(embeddings, expected_count=total, context="chunk encoding")

        logger.info(
            f"Chunk encoding complete | count={total} | "
            f"dim={len(embeddings[0])} | time={elapsed_ms:.1f}ms | "
            f"avg_per_chunk={elapsed_ms / total:.2f}ms",
            extra={"ai_pipeline": True},
        )

        return embeddings

    def encode_query(self, query: str) -> List[float]:
        """
        Generate a single embedding vector for a user query string.

        The returned vector is used to search ChromaDB for nearest neighbors.
        The same model that encoded the document chunks MUST encode the query —
        mixing models would make similarity scores meaningless.

        Args:
            query: The user's question or search string.

        Returns:
            A 384-dimensional float vector.

        Raises:
            ValueError: If query is empty or whitespace-only.
            RuntimeError: If embedding generation fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        logger.debug(
            f"Encoding query | length={len(query)} chars",
            extra={"ai_pipeline": True},
        )

        start = time.perf_counter()
        embeddings = self._encode_texts([query])
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._validate(embeddings, expected_count=1, context="query encoding")

        logger.debug(
            f"Query encoding complete | dim={len(embeddings[0])} | "
            f"time={elapsed_ms:.2f}ms",
            extra={"ai_pipeline": True},
        )

        # Unwrap the single-element list — callers expect List[float]
        return embeddings[0]

    # ── Observability ─────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded model.
        Used by the /health endpoint and logging diagnostics.
        """
        return {
            "model_name": self._model_name,
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": self._batch_size,
            "load_time_ms": round(self._load_time_ms, 1),
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Run SentenceTransformer inference with batch processing.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            List of embedding vectors as native Python float lists.
        """
        assert self._model is not None, "Model not loaded — call _load_model() first"

        try:
            # convert_to_numpy=True is the default and fastest path on CPU.
            # show_progress_bar=False keeps logs clean in production.
            numpy_embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
            )
        except Exception as e:
            logger.error(
                f"Embedding inference failed | error={e}",
                extra={"ai_pipeline": True},
            )
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Convert numpy → Python list for serialization safety
        # numpy arrays are not JSON-serializable and ChromaDB expects lists
        return numpy_embeddings.tolist()

    def _validate(
        self,
        embeddings: List[List[float]],
        expected_count: int,
        context: str,
    ) -> None:
        """
        Assert embedding output meets dimension and count contracts.

        Args:
            embeddings: Output from _encode_texts().
            expected_count: How many embeddings we expected.
            context: Human-readable context for error messages.
        """
        if len(embeddings) != expected_count:
            raise RuntimeError(
                f"[{context}] Embedding count mismatch: "
                f"expected {expected_count}, got {len(embeddings)}"
            )

        for i, vec in enumerate(embeddings):
            if len(vec) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"[{context}] Embedding #{i} has wrong dimension: "
                    f"expected {EMBEDDING_DIM}, got {len(vec)}"
                )

            if not vec:
                raise RuntimeError(
                    f"[{context}] Embedding #{i} is empty."
                )


# ── Singleton Access ──────────────────────────────────────────────────────────

def get_embedding_engine() -> EmbeddingEngine:
    """
    Return the application-wide EmbeddingEngine singleton.

    Thread-safe double-checked locking ensures the model is initialized
    exactly once, even if multiple async FastAPI workers call this
    concurrently during startup.

    Usage:
        engine = get_embedding_engine()
        query_vec = engine.encode_query("What is the main contribution?")

    This is the ONLY way to obtain an EmbeddingEngine in application code.
    Direct instantiation (EmbeddingEngine()) is reserved for tests.
    """
    global _instance

    # Fast path: no lock needed if already initialized (99.9% of requests)
    if _instance is not None:
        return _instance

    # Slow path: acquire lock, initialize once
    with _instance_lock:
        if _instance is None:  # Double-checked: another thread may have won
            logger.info("Initializing EmbeddingEngine singleton...")
            _instance = EmbeddingEngine()

    return _instance


def reset_embedding_engine() -> None:
    """
    Reset the singleton to None.

    ONLY for use in tests — allows test isolation by forcing a fresh
    engine instance. Never call this in production code.
    """
    global _instance
    with _instance_lock:
        _instance = None
