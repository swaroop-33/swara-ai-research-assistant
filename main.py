"""
main.py
========
FastAPI application entry point.

This module:
  1. Creates the FastAPI application instance
  2. Sets up logging at startup
  3. Registers all API routers
  4. Creates required runtime directories
  5. Exposes the app for Uvicorn to serve

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger

# Import routers (registered after app creation)
from app.api.routes import health, upload, query

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Runs startup logic before yielding, teardown after.
    This is the FastAPI-recommended pattern (replaces deprecated @on_event).
    """
    # --- Startup ---
    setup_logging()
    logger.info("AI Research Assistant starting up...")

    # Ensure required runtime directories exist
    for directory in ["uploads", "logs", "chroma_db", "data", "screenshots"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}/")

    logger.info(
        f"Configuration loaded | "
        f"env={settings.environment} | "
        f"primary_model={settings.groq_primary_model} | "
        f"embedding_model={settings.embedding_model}"
    )

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("AI Research Assistant shutting down gracefully.")


def create_application() -> FastAPI:
    """
    Application factory pattern.
    Returns a fully configured FastAPI instance.
    Using a factory makes it easy to create test instances with different config.
    """
    app = FastAPI(
        title="AI Research Assistant",
        description=(
            "A Retrieval-Augmented Generation (RAG) system for document-aware "
            "AI Q&A. Upload PDFs or text, ask questions, get contextual answers."
        ),
        version="1.0.0",
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI
        lifespan=lifespan,
    )

    # --- CORS ---
    # Allows the Streamlit frontend (running on port 8501) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register Routers ---
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
