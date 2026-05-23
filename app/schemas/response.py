"""
app/schemas/response.py
========================
Unified response schemas for the SWARA RAG pipeline.

These schemas define:
- retrieval transparency
- API response contracts
- frontend compatibility
- observability metadata
- diagnostics telemetry
"""

from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# =========================================================
# RETRIEVED CHUNK
# =========================================================

class RetrievedChunk(BaseModel):
    """
    Single retrieved chunk with transparency metadata.
    """

    chunk_id: str

    text: str

    filename: str

    page_number: Optional[int] = None

    document_type: str

    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic similarity score",
    )

    rank: int = Field(
        description="Retrieval ranking position",
    )

    timestamp: str = Field(
        description="ISO timestamp when chunk was indexed",
    )


# =========================================================
# RETRIEVAL DIAGNOSTICS
# =========================================================

class RetrievalDiagnostics(BaseModel):
    """
    Retrieval telemetry + observability metadata.
    """

    original_query: str

    expanded_query: str

    similarity_threshold: float

    retrieved_chunk_count: int

    filtered_chunk_count: int

    top_similarity_score: Optional[float] = None

    average_similarity_score: Optional[float] = None

    embedding_time_ms: float

    search_time_ms: float

    retrieval_time_ms: float

    factual_query_detected: bool


# =========================================================
# MAIN RAG RESPONSE
# =========================================================

class RAGResponse(BaseModel):
    """
    Full grounded response returned by the RAG pipeline.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
    )

    answer: str = Field(
        description="Grounded synthesized answer",
    )

    question: str = Field(
        description="Original user query",
    )

    rewritten_query: str = Field(
        description="Conversationally expanded query",
    )

    retrieved_chunks: List[RetrievedChunk] = Field(
        description="Retrieved semantic chunks",
    )

    model_used: str = Field(
        description="LLM model used for generation",
    )

    retrieval_time_ms: float

    generation_time_ms: float

    total_time_ms: float

    fallback_used: bool = Field(
        default=False,
        description="Whether fallback model was used",
    )

    diagnostics: RetrievalDiagnostics


# =========================================================
# STANDARD ERROR RESPONSE
# =========================================================

class ErrorResponse(BaseModel):
    """
    Standard API error schema.
    """

    error: str

    detail: Optional[str] = None

    code: int


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

QueryResponse = RAGResponse