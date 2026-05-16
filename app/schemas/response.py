"""
app/schemas/response.py
========================
Unified response schemas for the RAG pipeline.

These schemas define:
- retrieval transparency
- API response contracts
- frontend response structure
"""

from typing import List, Optional

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, Field


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
        description="Cosine similarity score",
    )

    rank: int = Field(
        description="Retrieval rank",
    )

    timestamp: str = Field(
        description="ISO timestamp when chunk was indexed",
    )


class RAGResponse(BaseModel):
    """
    Full response returned by the RAG pipeline.
    """

    # Prevent pydantic protected namespace warning
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    answer: str = Field(
        description="LLM-generated grounded answer",
    )

    question: str = Field(
        description="Original user query",
    )

    retrieved_chunks: List[RetrievedChunk] = Field(
        description="Retrieved context chunks",
    )

    model_used: str = Field(
        description="Groq model used for generation",
    )

    retrieval_time_ms: float

    generation_time_ms: float

    total_time_ms: float

    fallback_used: bool = Field(
        default=False,
        description="Whether fallback model was used",
    )


class ErrorResponse(BaseModel):
    """
    Standard API error schema.
    """

    error: str

    detail: Optional[str] = None

    code: int


# Backward compatibility alias
QueryResponse = RAGResponse
