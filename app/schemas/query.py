"""
app/schemas/query.py
=====================
Request schemas for SWARA query endpoints.
"""

from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
)


# =========================================================
# CHAT MESSAGE
# =========================================================

class ChatMessage(BaseModel):

    role: str = Field(
        description="Message role",
        examples=["user"],
    )

    content: str = Field(
        description="Message content",
    )


# =========================================================
# QUERY REQUEST
# =========================================================

class QueryRequest(BaseModel):
    """
    Query request payload for RAG pipeline.
    """

    question: str = Field(
        min_length=1,
        description="User question",
    )

    top_k: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of retrieved chunks",
    )

    chat_history: Optional[
        List[ChatMessage]
    ] = Field(
        default_factory=list,
        description="Recent conversation history",
    )