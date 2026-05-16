
"""
app/core/config.py
==================
Centralized configuration management using Pydantic BaseSettings.

Engineering rationale:
  - All environment variables are validated at startup
  - Type coercion is automatic
  - Single source of truth for config
  - IDE autocomplete across the project
"""

from functools import lru_cache

# pyrefly: ignore [missing-import]
from pydantic import Field
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────
    # Environment
    # ─────────────────────────────────────────────

    environment: str = Field(default="development")

    # ─────────────────────────────────────────────
    # FastAPI
    # ─────────────────────────────────────────────

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)

    # ─────────────────────────────────────────────
    # Uploads
    # ─────────────────────────────────────────────

    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded documents",
    )

    # ─────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────

    log_level: str = Field(default="INFO")

    log_dir: str = Field(
        default="./logs",
        description="Directory for runtime logs",
    )

    # ─────────────────────────────────────────────
    # Embedding
    # ─────────────────────────────────────────────

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer embedding model",
    )

    # ─────────────────────────────────────────────
    # ChromaDB
    # ─────────────────────────────────────────────

    chroma_persist_dir: str = Field(
        default="./chroma_db",
    )

    chroma_collection_name: str = Field(
        default="research_documents",
    )

    # ─────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────

    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
    )

    chunk_overlap: int = Field(
        default=80,
        ge=0,
        le=200,
    )

    # ─────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────

    top_k_results: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # ─────────────────────────────────────────────
    # Groq
    # ─────────────────────────────────────────────

    groq_api_key: str = Field(
        default="",
        description="Groq API key",
    )

    groq_primary_model: str = Field(
        default="llama3-70b-8192",
    )

    groq_fallback_model: str = Field(
        default="llama3-8b-8192",
    )

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def fastapi_backend_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.

    .env is loaded once per process lifetime.
    """
    return Settings()


# Singleton instance
settings = get_settings()
