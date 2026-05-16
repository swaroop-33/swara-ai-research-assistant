from pydantic import BaseModel, Field
from typing import List, Dict


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=10)

    # NEW: conversation memory
    chat_history: List[Dict] = []