"""
tests/test_embedder.py
=======================
Unit and integration tests for the EmbeddingEngine.

Testing philosophy:
  - These tests run against the REAL model (no mocking).
  - Mocking an embedding model would test nothing meaningful — we need to
    verify actual dimensional output, determinism, and semantic behavior.
  - The first run downloads model weights (~90MB) to HuggingFace cache.
    Subsequent runs are fast because the cache is reused.
  - We reset the singleton between test classes to ensure isolation.

Test categories:
  1. Basic functionality — encode chunks and queries successfully
  2. Dimension validation — 384-dim output contract
  3. Determinism — same input → identical vectors every time
  4. Semantic similarity — similar sentences score higher than unrelated ones
  5. Batch processing — large inputs processed stably
  6. Singleton behavior — get_embedding_engine() returns the same instance
  7. Input validation — empty inputs raise correct errors
"""

import math
import pytest

from app.document_processor import DocumentChunk, ExtractionResult, chunk_document
from embeddings.embedder import (
    EMBEDDING_DIM,
    EmbeddingEngine,
    get_embedding_engine,
    reset_embedding_engine,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_chunk(text: str, index: int = 0, filename: str = "test.txt") -> DocumentChunk:
    """Build a minimal DocumentChunk for embedding tests."""
    from datetime import datetime, timezone
    return DocumentChunk(
        chunk_id=f"test_chunk_{index:04d}",
        text=text,
        filename=filename,
        document_type="txt",
        page_number=None,
        chunk_index=index,
        char_count=len(text),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Since our embeddings are L2-normalized, this equals the dot product.
    Kept explicit for test readability.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """
    Create a single EmbeddingEngine for the entire test module.
    scope="module" means the model loads once for all tests in this file —
    not once per test — matching production singleton behavior.
    """
    reset_embedding_engine()
    eng = EmbeddingEngine()
    yield eng
    reset_embedding_engine()


# ─────────────────────────────────────────────
# 1. Basic Functionality
# ─────────────────────────────────────────────

class TestBasicEncoding:
    """Verify encode_chunks and encode_query return the right types and sizes."""

    def test_encode_single_chunk_returns_list(self, engine):
        chunk = _make_chunk("Machine learning is a subfield of artificial intelligence.")
        result = engine.encode_chunks([chunk])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_encode_multiple_chunks_returns_correct_count(self, engine):
        chunks = [_make_chunk(f"Sentence number {i}.", index=i) for i in range(5)]
        result = engine.encode_chunks(chunks)
        assert len(result) == 5

    def test_encode_query_returns_single_list(self, engine):
        result = engine.encode_query("What is retrieval-augmented generation?")
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_encode_query_is_not_nested(self, engine):
        """encode_query must return List[float], NOT List[List[float]]."""
        result = engine.encode_query("test query")
        # If result[0] were a list, this would fail
        assert isinstance(result[0], float), (
            "encode_query should return List[float], not List[List[float]]"
        )

    def test_output_contains_only_floats(self, engine):
        chunk = _make_chunk("Neural networks learn from data.")
        embeddings = engine.encode_chunks([chunk])
        for val in embeddings[0]:
            assert isinstance(val, float)


# ─────────────────────────────────────────────
# 2. Dimension Validation
# ─────────────────────────────────────────────

class TestEmbeddingDimensions:
    """Verify the 384-dimensional output contract is never violated."""

    def test_chunk_embedding_is_384_dimensional(self, engine):
        chunk = _make_chunk("Transformers are powerful sequence models.")
        result = engine.encode_chunks([chunk])
        assert len(result[0]) == EMBEDDING_DIM, (
            f"Expected {EMBEDDING_DIM} dimensions, got {len(result[0])}"
        )

    def test_query_embedding_is_384_dimensional(self, engine):
        result = engine.encode_query("What are the key findings?")
        assert len(result) == EMBEDDING_DIM

    def test_all_chunks_have_same_dimension(self, engine):
        texts = [
            "Short sentence.",
            "A much longer sentence with many more words and detailed context.",
            "Medium length sentence about machine learning systems.",
        ]
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        embeddings = engine.encode_chunks(chunks)
        dims = [len(e) for e in embeddings]
        assert len(set(dims)) == 1, f"Inconsistent embedding dimensions: {dims}"
        assert dims[0] == EMBEDDING_DIM

    def test_embedding_dimension_constant_matches_model(self, engine):
        """The EMBEDDING_DIM constant must match actual model output."""
        result = engine.encode_query("verify constant matches reality")
        assert len(result) == EMBEDDING_DIM


# ─────────────────────────────────────────────
# 3. Determinism
# ─────────────────────────────────────────────

class TestDeterminism:
    """Verify same input always produces identical output vectors."""

    def test_chunk_encoding_is_deterministic(self, engine):
        chunk = _make_chunk("The attention mechanism is key to transformer performance.")
        emb_a = engine.encode_chunks([chunk])[0]
        emb_b = engine.encode_chunks([chunk])[0]
        assert emb_a == emb_b, "Embedding must be identical across two calls with same input"

    def test_query_encoding_is_deterministic(self, engine):
        query = "How does BERT use bidirectional context?"
        emb_a = engine.encode_query(query)
        emb_b = engine.encode_query(query)
        assert emb_a == emb_b

    def test_ordering_within_batch_is_deterministic(self, engine):
        """Embeddings[i] must always correspond to chunks[i], batch after batch."""
        texts = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]

        run1 = engine.encode_chunks(chunks)
        run2 = engine.encode_chunks(chunks)

        for i in range(len(chunks)):
            assert run1[i] == run2[i], f"Chunk {i} embedding changed between runs"

    def test_single_vs_batch_encoding_produces_same_vector(self, engine):
        """
        Encoding a chunk alone vs. in a batch must produce the same vector.
        This verifies the batch processing doesn't perturb individual results.
        """
        chunk_a = _make_chunk("Knowledge graphs represent structured information.")
        chunk_b = _make_chunk("Irrelevant sentence about cooking pasta.")

        # Encode individually
        solo_emb = engine.encode_chunks([chunk_a])[0]

        # Encode as part of a batch
        batch_embs = engine.encode_chunks([chunk_a, chunk_b])
        batch_emb = batch_embs[0]

        # Due to L2 normalization these should be identical
        # (normalize_embeddings=True makes batch behavior consistent)
        similarity = cosine_similarity(solo_emb, batch_emb)
        assert similarity > 0.9999, (
            f"Solo vs batch embedding diverged: similarity={similarity:.6f}"
        )


# ─────────────────────────────────────────────
# 4. Semantic Similarity
# ─────────────────────────────────────────────

class TestSemanticSimilarity:
    """
    Verify that the embedding space captures semantic meaning.
    This is the most important test for a RAG system — it validates
    that our retrieval will actually find relevant chunks.
    """

    def test_similar_sentences_have_higher_similarity_than_unrelated(self, engine):
        """
        Core RAG quality test.
        Semantically similar pairs must score higher than unrelated pairs.
        """
        # Semantically similar
        emb_a = engine.encode_query(
            "Deep learning is a branch of machine learning."
        )
        emb_b = engine.encode_query(
            "Neural networks are used in deep learning research."
        )

        # Semantically unrelated
        emb_c = engine.encode_query(
            "The recipe calls for two cups of flour and one egg."
        )

        sim_related = cosine_similarity(emb_a, emb_b)
        sim_unrelated = cosine_similarity(emb_a, emb_c)

        assert sim_related > sim_unrelated, (
            f"Semantic similarity test failed: "
            f"related={sim_related:.4f} should be > unrelated={sim_unrelated:.4f}"
        )

    def test_identical_sentences_have_near_perfect_similarity(self, engine):
        """Same sentence → cosine similarity ≈ 1.0 (perfect match)."""
        text = "Retrieval-augmented generation improves factual accuracy."
        emb_a = engine.encode_query(text)
        emb_b = engine.encode_query(text)
        sim = cosine_similarity(emb_a, emb_b)
        assert sim > 0.9999, f"Identical sentences should score ~1.0, got {sim:.6f}"

    def test_paraphrase_scores_higher_than_random(self, engine):
        """A paraphrase of a sentence should be more similar than random text."""
        original = engine.encode_query(
            "The model was trained on a large corpus of scientific papers."
        )
        paraphrase = engine.encode_query(
            "Training data consisted of many academic research documents."
        )
        random = engine.encode_query(
            "Monday is the first day of the working week."
        )

        sim_paraphrase = cosine_similarity(original, paraphrase)
        sim_random = cosine_similarity(original, random)

        assert sim_paraphrase > sim_random, (
            f"Paraphrase ({sim_paraphrase:.4f}) should be more similar "
            f"than random text ({sim_random:.4f})"
        )

    def test_query_and_relevant_chunk_have_high_similarity(self, engine):
        """
        Simulate a real RAG retrieval scenario.
        A question and an answer-containing chunk should have high similarity.
        """
        query_emb = engine.encode_query(
            "What are transformers used for in NLP?"
        )
        relevant_chunk = _make_chunk(
            "Transformers are the dominant architecture in NLP, "
            "used for tasks like machine translation, summarization, "
            "and question answering due to their self-attention mechanism."
        )
        irrelevant_chunk = _make_chunk(
            "The electrical transformer steps up or steps down voltage "
            "in power distribution networks using electromagnetic induction."
        )

        relevant_emb = engine.encode_chunks([relevant_chunk])[0]
        irrelevant_emb = engine.encode_chunks([irrelevant_chunk])[0]

        sim_relevant = cosine_similarity(query_emb, relevant_emb)
        sim_irrelevant = cosine_similarity(query_emb, irrelevant_emb)

        assert sim_relevant > sim_irrelevant, (
            f"Relevant chunk ({sim_relevant:.4f}) should score higher "
            f"than irrelevant chunk ({sim_irrelevant:.4f})"
        )


# ─────────────────────────────────────────────
# 5. Batch Processing Stability
# ─────────────────────────────────────────────

class TestBatchProcessing:
    """Verify large batches are handled efficiently and correctly."""

    def test_large_batch_produces_correct_count(self, engine):
        """50 chunks should produce exactly 50 embeddings."""
        chunks = [_make_chunk(f"Research document sentence {i}.", i) for i in range(50)]
        result = engine.encode_chunks(chunks)
        assert len(result) == 50

    def test_batch_larger_than_batch_size_works(self, engine):
        """
        Batches larger than batch_size (32) must be handled correctly.
        This tests that our batch_size parameter splits large inputs.
        """
        chunks = [_make_chunk(f"Batch test sentence {i}.", i) for i in range(80)]
        result = engine.encode_chunks(chunks)
        assert len(result) == 80
        for emb in result:
            assert len(emb) == EMBEDDING_DIM

    def test_single_chunk_batch_works(self, engine):
        """Batch of 1 should not fail or produce wrong output."""
        chunk = _make_chunk("A single sentence.", 0)
        result = engine.encode_chunks([chunk])
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM


# ─────────────────────────────────────────────
# 6. Singleton Behavior
# ─────────────────────────────────────────────

class TestSingleton:
    """Verify get_embedding_engine() returns a shared instance."""

    def test_get_engine_returns_same_instance_across_calls(self):
        """Two calls to get_embedding_engine() must return the SAME object."""
        reset_embedding_engine()
        engine_a = get_embedding_engine()
        engine_b = get_embedding_engine()
        assert engine_a is engine_b, (
            "Singleton violated — two different EmbeddingEngine instances were created"
        )
        reset_embedding_engine()

    def test_singleton_has_correct_model_name(self):
        reset_embedding_engine()
        engine = get_embedding_engine()
        info = engine.get_model_info()
        assert "MiniLM" in info["model_name"] or "minilm" in info["model_name"].lower()
        reset_embedding_engine()

    def test_reset_allows_fresh_instantiation(self):
        """reset_embedding_engine() must actually clear the singleton."""
        engine_a = get_embedding_engine()
        reset_embedding_engine()
        engine_b = get_embedding_engine()
        # After reset, a new object should be created
        assert engine_a is not engine_b
        reset_embedding_engine()


# ─────────────────────────────────────────────
# 7. Input Validation
# ─────────────────────────────────────────────

class TestInputValidation:
    """Verify invalid inputs raise clear, descriptive errors."""

    def test_empty_chunk_list_raises_value_error(self, engine):
        with pytest.raises(ValueError, match="empty"):
            engine.encode_chunks([])

    def test_empty_query_raises_value_error(self, engine):
        with pytest.raises(ValueError):
            engine.encode_query("")

    def test_whitespace_only_query_raises_value_error(self, engine):
        with pytest.raises(ValueError):
            engine.encode_query("   ")

    def test_get_model_info_returns_expected_keys(self, engine):
        info = engine.get_model_info()
        assert "model_name" in info
        assert "embedding_dim" in info
        assert "batch_size" in info
        assert "load_time_ms" in info
        assert info["embedding_dim"] == EMBEDDING_DIM
