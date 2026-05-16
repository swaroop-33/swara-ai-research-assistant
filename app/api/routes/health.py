"""
app/api/routes/health.py
=========================
Health check endpoint.

A health endpoint is standard in production services — it's used by:
  - Load balancers to verify the service is alive
  - Deployment platforms (Render, Railway) to check readiness
  - Monitoring systems to track uptime

For a portfolio project, having /health shows you understand
production deployment concerns.
"""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check() -> dict:
    """
    Returns application health status and basic configuration info.
    Used for deployment readiness checks.
    """
    return {
        "status": "healthy",
        "service": "AI Research Assistant",
        "version": "1.0.0",
        "environment": settings.environment,
        "embedding_model": settings.embedding_model,
        "primary_llm": settings.groq_primary_model,
    }
