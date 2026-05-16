"""
tests/test_retrieval.py
========================
Tests for the VectorStore (ChromaDB) layer.

Testing strategy:
  - Every test gets its own isolated ChromaDB instance via tmp_path.
    This guarantees tests never share state and are safe to run in any order.
  - Basic CRUD tests (insert, query, clear, stats) use synthetic 384-dim
    vectors — fast, no model loading required.
  - The semantic ranking test uses real embeddings from EmbeddingEngine to
    verify that ChromaDB actually returns MORE relevant chunks first.
  - We test the full contract: RetrievedChunk schema, similarity scores,
    rank ordering, metadata round-trip, and persistence.

Why test with tmp_path?
  - If we used the real chroma_db/ directory, tests would pollute each other
    and production data. tmp_path is automatically cleaned up by pytest.
"""

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from app.document_processor import DocumentChunk
from app.schemas.response import RetrievedChunk
from vectorstore.chroma_store import AddResult, CollectionStats, VectorStore


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_store(tmp_path: Path, suffix: str = "") -> VectorStore:
    """
    Create an isolated VectorStore in tmp_path.
    Each call with a different suffix creates a distinct collection.
    """
    collection_name = f"test_col_{suffix or uuid.uuid4().hex[:8]}"
    return VectorStore(
        persist_dir=str(tmp_path / "chroma_db"),
        collection_name=collection_name,
    )


def _make_chunk(
    text: str,
    index: int = 0,
    filename: str = "doc.txt",
    page: int | None = None,
    doc_type: str = "txt",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{filename.split('.')[0]}_chunk_{index:04d}",
        text=text,
        filename=filename,
        document_type=doc_type,
        page_number=page,
        chunk_index=index,
        char_count=len(text),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _unit_vector(dim: int = 384, seed_val: float = 0.1) -> List[float]:
    """
    Create a synthetic L2-normalized vector for testing.
    Using a constant value produces a valid unit vector along the uniform direction.
    """
    raw = [seed_val] * dim
    norm = math.sqrt(sum(x ** 2 for x in raw))
    return [x / norm for x in raw]


def _orthogonal_vector(dim: int = 384) -> List[float]:
    """
    Create a vector orthogonal to _unit_vector (cosine similarity = 0).
    Alternates +val / -val so dot product with uniform vector = 0.
    """
    val = 1.0 / math.sqrt(dim)
    raw = [val if i % 2 == 0 else -val for i in range(dim)]
    norm = math.sqrt(sum(x ** 2 for x in raw))
    return [x / norm for x in raw]


# ─────────────────────────────────────────────
# 1. Initialization & Stats
# ─────────────────────────────────────────────

class TestInitializationAndStats:

    def test_store_initializes_cleanly(self, tmp_path):
        store = _make_store(tmp_path)
        stats = store.get_stats()
        assert stats.is_ready is True

    def test_empty_collection_has_zero_chunks(self, tmp_path):
        store = _make_store(tmp_path)
        stats = store.get_stats()
        assert stats.total_chunks == 0

    def test_stats_returns_correct_collection_name(self, tmp_path):
        store = _make_store(tmp_path, suffix="named")
        stats = store.get_stats()
        assert "named" in stats.collection_name

    def test_stats_persist_directory_is_set(self, tmp_path):
        store = _make_store(tmp_path)
        stats = store.get_stats()
        assert stats.persist_directory != ""

    def test_chroma_db_directory_created_on_init(self, tmp_path):
        _make_store(tmp_path)
        assert (tmp_path / "chroma_db").exists()


# ─────────────────────────────────────────────
# 2. Document Insertion
# ─────────────────────────────────────────────

class TestDocumentInsertion:

    def test_add_single_chunk_succeeds(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Hello world.", index=0)
        embedding = _unit_vector()

        result = store.add_documents([chunk], [embedding])

        assert result.success is True
        assert result.chunks_added == 1

    def test_add_multiple_chunks_updates_count(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(f"Chunk {i}.", index=i) for i in range(5)]
        embeddings = [_unit_vector(seed_val=0.1 + i * 0.01) for i in range(5)]

        result = store.add_documents(chunks, embeddings)

        assert result.success is True
        assert result.chunks_added == 5
        assert store.get_stats().total_chunks == 5

    def test_add_returns_insertion_time(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Timing test.")
        result = store.add_documents([chunk], [_unit_vector()])
        assert result.insertion_time_ms >= 0

    def test_mismatched_chunks_embeddings_returns_failure(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk("A"), _make_chunk("B")]
        embeddings = [_unit_vector()]  # Only 1 embedding for 2 chunks

        result = store.add_documents(chunks, embeddings)

        assert result.success is False
        assert result.error is not None

    def test_empty_chunks_returns_failure(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.add_documents([], [])
        assert result.success is False

    def test_upsert_same_id_does_not_duplicate(self, tmp_path):
        """Adding the same chunk twice should update, not duplicate."""
        store = _make_store(tmp_path)
        chunk = _make_chunk("Original text.")

        store.add_documents([chunk], [_unit_vector()])
        store.add_documents([chunk], [_unit_vector()])  # same chunk_id

        # Should still be 1 chunk, not 2
        assert store.get_stats().total_chunks == 1


# ─────────────────────────────────────────────
# 3. Metadata Persistence
# ─────────────────────────────────────────────

class TestMetadataPersistence:

    def test_filename_is_preserved(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Text.", filename="my_paper.pdf", doc_type="pdf", page=3)
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].filename == "my_paper.pdf"

    def test_document_type_is_preserved(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Text.", doc_type="pdf", page=1)
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].document_type == "pdf"

    def test_page_number_is_preserved_for_pdf(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("PDF content.", doc_type="pdf", page=7)
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].page_number == 7

    def test_none_page_number_is_restored_for_txt(self, tmp_path):
        """
        TXT chunks have page_number=None.
        This is stored as sentinel -1 in ChromaDB and must be restored to None on read.
        """
        store = _make_store(tmp_path)
        chunk = _make_chunk("TXT content.", doc_type="txt", page=None)
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].page_number is None

    def test_timestamp_is_preserved(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Timestamp test.")
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].timestamp != ""
        # Verify it's valid ISO 8601
        datetime.fromisoformat(results[0].timestamp)

    def test_chunk_id_matches_original(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("ID test.", index=42)
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert results[0].chunk_id == chunk.chunk_id


# ─────────────────────────────────────────────
# 4. Query Behavior
# ─────────────────────────────────────────────

class TestQueryBehavior:

    def test_query_empty_collection_returns_empty_list(self, tmp_path):
        store = _make_store(tmp_path)
        results = store.query(_unit_vector(), n_results=5)
        assert results == []

    def test_query_returns_retrieved_chunk_objects(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk("Some content.")
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=1)

        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_query_respects_n_results(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(f"Chunk {i}.", index=i) for i in range(10)]
        embeddings = [_unit_vector(seed_val=0.1 + i * 0.01) for i in range(10)]
        store.add_documents(chunks, embeddings)

        results = store.query(_unit_vector(), n_results=3)

        assert len(results) == 3

    def test_query_n_results_clamped_to_collection_size(self, tmp_path):
        """Requesting more results than stored chunks should not crash."""
        store = _make_store(tmp_path)
        chunk = _make_chunk("Only one chunk.")
        store.add_documents([chunk], [_unit_vector()])

        results = store.query(_unit_vector(), n_results=100)

        assert len(results) == 1  # Clamped to actual count

    def test_similarity_scores_are_in_valid_range(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(f"Content {i}.", index=i) for i in range(3)]
        embeddings = [_unit_vector(seed_val=0.1 + i * 0.05) for i in range(3)]
        store.add_documents(chunks, embeddings)

        results = store.query(_unit_vector(), n_results=3)

        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_ranks_are_sequential_starting_at_one(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(f"Chunk {i}.", index=i) for i in range(5)]
        embeddings = [_unit_vector(seed_val=0.1 + i * 0.01) for i in range(5)]
        store.add_documents(chunks, embeddings)

        results = store.query(_unit_vector(), n_results=5)

        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_identical_query_vector_gets_high_similarity(self, tmp_path):
        """A chunk embedded with the same vector as the query should score ~1.0."""
        store = _make_store(tmp_path)
        vec = _unit_vector(seed_val=0.5)
        chunk = _make_chunk("Identical vector chunk.")
        store.add_documents([chunk], [vec])

        results = store.query(vec, n_results=1)

        assert results[0].similarity_score > 0.99

    def test_orthogonal_vector_gets_low_similarity(self, tmp_path):
        """A chunk embedded with an orthogonal vector to the query scores ~0.5."""
        store = _make_store(tmp_path)
        chunk = _make_chunk("Orthogonal vector chunk.")
        store.add_documents([chunk], [_unit_vector(seed_val=0.1)])

        # Query with orthogonal vector — should score near 0.5 (distance ≈ 1.0)
        results = store.query(_orthogonal_vector(), n_results=1)

        assert results[0].similarity_score < 0.6


# ─────────────────────────────────────────────
# 5. Semantic Ranking (Integration)
# ─────────────────────────────────────────────

class TestSemanticRanking:
    """
    Integration test: verify ChromaDB returns semantically relevant
    chunks above irrelevant ones using REAL embeddings.

    This test validates the complete Phase 3→4 pipeline boundary:
    EmbeddingEngine → VectorStore → RetrievedChunk with correct ranking.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        from embeddings.embedder import EmbeddingEngine, reset_embedding_engine
        reset_embedding_engine()
        eng = EmbeddingEngine()
        yield eng
        reset_embedding_engine()

    def test_relevant_chunk_ranks_above_irrelevant(self, tmp_path, engine):
        """
        Core RAG pipeline validation:
        A question about transformers in NLP should retrieve the NLP chunk
        ranked above the electrical transformer chunk.
        """
        store = _make_store(tmp_path, suffix="semantic")

        # Build chunks with contrasting content
        nlp_chunk = _make_chunk(
            "Transformers in natural language processing use self-attention "
            "mechanisms to capture long-range dependencies in text sequences. "
            "BERT and GPT are prominent transformer-based language models.",
            index=0,
            filename="nlp_paper.txt",
        )
        electrical_chunk = _make_chunk(
            "Electrical transformers are devices that transfer electrical energy "
            "between circuits through electromagnetic induction. They are used "
            "in power distribution to step up or step down voltage levels.",
            index=1,
            filename="electrical.txt",
        )
        cooking_chunk = _make_chunk(
            "To make pasta carbonara, cook spaghetti until al dente. "
            "Mix eggs and parmesan cheese. Combine with pancetta.",
            index=2,
            filename="recipes.txt",
        )

        # Generate real embeddings
        embeddings = engine.encode_chunks([nlp_chunk, electrical_chunk, cooking_chunk])
        store.add_documents(
            [nlp_chunk, electrical_chunk, cooking_chunk],
            embeddings,
        )

        # Query about NLP transformers
        query_emb = engine.encode_query(
            "How do transformer models work in natural language processing?"
        )
        results = store.query(query_emb, n_results=3)

        assert len(results) == 3

        # The NLP chunk must rank #1
        top_result = results[0]
        assert top_result.filename == "nlp_paper.txt", (
            f"Expected NLP chunk at rank 1, got '{top_result.filename}' "
            f"(score={top_result.similarity_score:.4f})"
        )

        # NLP chunk must score higher than cooking chunk
        nlp_score = next(r.similarity_score for r in results if r.filename == "nlp_paper.txt")
        cooking_score = next(r.similarity_score for r in results if r.filename == "recipes.txt")
        assert nlp_score > cooking_score

    def test_results_sorted_descending_by_similarity(self, tmp_path, engine):
        """Results must always be in descending similarity order."""
        store = _make_store(tmp_path, suffix="sorted")

        texts = [
            "Machine learning uses statistical methods to enable computers to learn.",
            "Deep learning is a subset of machine learning using neural networks.",
            "The weather today is sunny with mild temperatures.",
        ]
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        embeddings = engine.encode_chunks(chunks)
        store.add_documents(chunks, embeddings)

        query_emb = engine.encode_query("What is deep learning and machine learning?")
        results = store.query(query_emb, n_results=3)

        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by descending similarity: {scores}"
        )


# ─────────────────────────────────────────────
# 6. Collection Management
# ─────────────────────────────────────────────

class TestCollectionManagement:

    def test_clear_collection_removes_all_chunks(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(f"Chunk {i}.", index=i) for i in range(5)]
        embeddings = [_unit_vector() for _ in range(5)]
        store.add_documents(chunks, embeddings)

        store.clear_collection()

        assert store.get_stats().total_chunks == 0

    def test_can_add_after_clear(self, tmp_path):
        """Collection must be usable immediately after clearing."""
        store = _make_store(tmp_path)
        chunks = [_make_chunk("Before clear.", index=0)]
        store.add_documents(chunks, [_unit_vector()])

        store.clear_collection()

        new_chunk = _make_chunk("After clear.", index=0)
        result = store.add_documents([new_chunk], [_unit_vector()])

        assert result.success is True
        assert store.get_stats().total_chunks == 1

    def test_query_after_clear_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_documents([_make_chunk("Some text.")], [_unit_vector()])
        store.clear_collection()

        results = store.query(_unit_vector(), n_results=5)
        assert results == []


# ─────────────────────────────────────────────
# 7. Persistence
# ─────────────────────────────────────────────

class TestPersistence:

    def test_data_persists_across_store_instances(self, tmp_path):
        """
        Write to one VectorStore instance, open a new instance pointing
        to the same path — data must still be there.
        This validates ChromaDB's on-disk persistence.
        """
        persist_dir = str(tmp_path / "persistent_db")
        collection_name = "persist_test"

        # Instance A: write data
        store_a = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        chunks = [_make_chunk(f"Persistent chunk {i}.", index=i) for i in range(3)]
        embeddings = [_unit_vector(seed_val=0.1 + i * 0.1) for i in range(3)]
        store_a.add_documents(chunks, embeddings)

        # Instance B: open same path — no writes, just read
        store_b = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        stats = store_b.get_stats()

        assert stats.total_chunks == 3, (
            f"Expected 3 persisted chunks, got {stats.total_chunks}. "
            "ChromaDB persistence is not working correctly."
        )
