"""
app/schemas/upload.py
======================
Pydantic schemas for document upload request/response contracts.
"""

from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    """Response returned after a successful document upload and processing."""

    success: bool
    message: str
    filename: str
    document_type: str = Field(description="pdf or txt")
    num_chunks: int = Field(description="Number of text chunks created")
    num_vectors_stored: int = Field(description="Vectors written to ChromaDB")
    processing_time_ms: float = Field(description="Total pipeline duration in ms")


class ProcessingStatus(BaseModel):
    """Current status of the vector store."""

    is_ready: bool
    num_documents: int
    num_chunks: int
    collection_name: str
