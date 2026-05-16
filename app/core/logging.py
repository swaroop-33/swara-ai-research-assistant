"""
app/core/logging.py
====================
Centralized logging setup using Loguru.

Why Loguru over Python's built-in logging?
  - Zero boilerplate — no handlers, formatters, or getLogger() calls.
  - Structured output: timestamps, log levels, and caller info out of the box.
  - File rotation built in: logs rotate daily and are compressed automatically.
  - Async-safe: works correctly with FastAPI's async request handlers.

Usage anywhere in the project:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Document processed", extra={"filename": "paper.pdf"})
"""

import sys
from pathlib import Path
from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """
    Configure Loguru for the application.
    Called once at application startup in main.py.
    """
    # Remove the default Loguru handler so we control the format completely
    logger.remove()

    # --- Console handler ---
    # Human-readable format for development
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        enqueue=True,  # thread-safe async logging
    )

    # --- File handler ---
    # Rotating daily logs, retained for 7 days, compressed
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        rotation="00:00",       # rotate at midnight
        retention="7 days",     # keep 7 days of logs
        compression="zip",      # compress old logs
        enqueue=True,
    )

    # --- AI-specific log file ---
    # Separate file for AI pipeline events (embedding, retrieval, LLM calls)
    logger.add(
        log_dir / "ai_pipeline_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        filter=lambda record: record["extra"].get("ai_pipeline", False),
        rotation="00:00",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )

    logger.info(
        f"Logging initialized | level={settings.log_level} | env={settings.environment}"
    )


def get_logger(name: str):
    """
    Return a contextualized logger bound to the given module name.
    This is the standard way to get a logger in any module.

    Example:
        logger = get_logger(__name__)
        logger.info("Ready")
    """
    return logger.bind(module=name)
