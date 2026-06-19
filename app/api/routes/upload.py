"""
app/api/routes/upload.py
=========================
Production-ready upload + vectorstore management routes.

Responsibilities:
- Upload documents
- Trigger ingestion pipeline
- Return ingestion statistics
- Manage vectorstore lifecycle

Routes MUST remain thin.
Business logic belongs in services/vectorstore.
"""

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingestion_service import IngestionService
from app.utils.file_utils import sanitize_filename
from vectorstore.chroma_store import VectorStore, get_vector_store

router = APIRouter()

logger = get_logger(__name__)

# Singletons are created lazily inside handlers — not at import time.
# This prevents the 90MB embedding model from loading during FastAPI boot.
_ingestion_service: IngestionService | None = None
_vectorstore: VectorStore | None = None


def _get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        # CRITICAL: pass the shared singleton, not a new VectorStore()
        _ingestion_service = IngestionService(
            vectorstore=get_vector_store()
        )
    return _ingestion_service


def _get_vectorstore() -> VectorStore:
    # Always return the shared singleton
    return get_vector_store()



@router.post(
    "/upload",
    summary="Upload and Process Document",
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """
    Upload a document and run the full ingestion pipeline.

    Flow:
        upload
        → save file
        → extraction
        → chunking
        → embedding
        → vector storage
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    # Sanitize filename to prevent path traversal attacks
    safe_filename = sanitize_filename(file.filename)
    if not safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )
        
    # Sanitize session_id to prevent path traversal attacks
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('_', '-'))
    if not safe_session_id:
        safe_session_id = "default"

    logger.info(
        f"Upload received | filename={safe_filename} | session_id={safe_session_id}",
        extra={"ai_pipeline": True},
    )

    # Store files in isolated session directories
    upload_dir = Path(settings.upload_dir) / safe_session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_filename

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ingestion pipeline
        result = _get_ingestion_service().ingest_file(
            file_path=file_path,
            filename=safe_filename,
            session_id=safe_session_id,
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error,
            )

        logger.info(
            f"Upload complete | "
            f"file={file.filename} | "
            f"session_id={safe_session_id} | "
            f"chunks={result.stats.total_chunks}",
            extra={"ai_pipeline": True},
        )

        return {
            "success": True,
            "filename": result.filename,
            "document_type": result.document_type,
            "chunks_created": result.stats.total_chunks,
            "embeddings_created": result.embeddings_created,
            "vectors_stored": result.vectors_stored,
            "processing_time_ms": round(result.processing_time_ms, 2),
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            f"Upload pipeline failed | error={e}",
            extra={"ai_pipeline": True},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload pipeline failed: {str(e)}",
        )


@router.get(
    "/stats",
    summary="Get Vector Store Statistics",
)
async def get_stats(
    session_id: str = "default"
):
    """
    Return vectorstore statistics for the given session.
    """

    try:
        stats = _get_vectorstore().get_stats(session_id=session_id)

        logger.info(
            f"Vectorstore stats requested | session_id={session_id}",
            extra={"ai_pipeline": True},
        )

        return stats

    except Exception as e:
        logger.exception(
            f"Stats retrieval failed | error={e}",
            extra={"ai_pipeline": True},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}",
        )


@router.delete(
    "/reset",
    summary="Reset Vector Store",
)
async def reset_store(
    session_id: str = "default"
):
    """
    Clear all stored vectors/documents for the given session.
    """

    try:
        success = _get_vectorstore().clear_collection(session_id=session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset vector store.",
            )

        logger.warning(
            f"Vectorstore reset performed | session_id={session_id}",
            extra={"ai_pipeline": True},
        )

        return {
            "success": True,
            "message": f"Vector store reset successfully for session: {session_id}",
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            f"Vectorstore reset failed | error={e}",
            extra={"ai_pipeline": True},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {str(e)}",
        )
