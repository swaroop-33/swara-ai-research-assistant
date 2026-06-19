"""
app/api/routes/query.py
========================
Production-ready query endpoint.
"""

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.core.logging import get_logger

from app.schemas.query import QueryRequest

from app.schemas.response import (
    RAGResponse,
)

from app.services.rag_service import (
    get_rag_service,
)

router = APIRouter()

logger = get_logger(__name__)


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Ask Questions About Uploaded Documents",
    status_code=status.HTTP_200_OK,
)
async def query_documents(
    request: QueryRequest,
) -> RAGResponse:
    """
    Query uploaded documents using SWARA RAG pipeline.
    """

    logger.info(
        f"Query received | "
        f"session_id={request.session_id} | "
        f"question='{request.question[:80]}'",
        extra={"ai_pipeline": True},
    )

    try:

        rag_service = get_rag_service()

        rag_response = rag_service.answer(
            question=request.question,
            n_results=request.top_k,
            chat_history=request.chat_history,
            session_id=request.session_id,
        )

        response = RAGResponse(
            answer=rag_response.answer,
            question=rag_response.question,
            rewritten_query=(
                rag_response.rewritten_query
            ),
            retrieved_chunks=(
                rag_response.retrieved_chunks
            ),
            model_used=rag_response.model_used,
            retrieval_time_ms=(
                rag_response.retrieval_time_ms
            ),
            generation_time_ms=(
                rag_response.generation_time_ms
            ),
            total_time_ms=(
                rag_response.total_time_ms
            ),
            fallback_used=(
                rag_response.fallback_used
            ),
            diagnostics=(
                rag_response.diagnostics
            ),
        )

        logger.info(
            f"Query completed | "
            f"session_id={request.session_id} | "
            f"chunks={len(response.retrieved_chunks)} | "
            f"model={response.model_used}",
            extra={"ai_pipeline": True},
        )

        return response

    except Exception as e:

        logger.exception(
            "Query pipeline failed",
            extra={"ai_pipeline": True},
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=f"Query pipeline failed: {str(e)}",
        )