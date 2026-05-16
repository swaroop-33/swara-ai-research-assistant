
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
    HTTPException,
    UploadFile,
    status,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingestion_service import IngestionService
from vectorstore.chroma_store import VectorStore

router = APIRouter()

logger = get_logger(__name__)

# Singletons are created lazily inside handlers — not at import time.
# This prevents the 90MB embedding model from loading during FastAPI boot.
_ingestion_service: IngestionService | None = None
_vectorstore: VectorStore | None = None


def _get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


def _get_vectorstore() -> VectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore



@router.post(
    "/upload",
    summary="Upload and Process Document",
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    file: UploadFile = File(...),
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

    logger.info(
        f"Upload received | filename={file.filename}",
        extra={"ai_pipeline": True},
    )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ingestion pipeline
        result = _get_ingestion_service().ingest_file(
            file_path=file_path,
            filename=file.filename,
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error,
            )

        logger.info(
            f"Upload complete | "
            f"file={file.filename} | "
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
async def get_stats():
    """
    Return vectorstore statistics.
    """

    try:
        stats = _get_vectorstore().get_stats()

        logger.info(
            "Vectorstore stats requested",
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
async def reset_store():
    """
    Clear all stored vectors/documents.
    """

    try:
        success = _get_vectorstore().clear_collection()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset vector store.",
            )

        logger.warning(
            "Vectorstore reset performed",
            extra={"ai_pipeline": True},
        )

        return {
            "success": True,
            "message": "Vector store reset successfully.",
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
