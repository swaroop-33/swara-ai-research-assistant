"""
app/utils/file_utils.py
========================
File system utilities for handling uploaded documents.

Responsibilities:
  - Saving uploaded files to the uploads/ directory safely
  - Generating safe filenames (sanitizing user input)
  - Cleaning up temporary files after processing
  - Validating file extensions before any processing occurs
"""

import re
import shutil
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Allowed extensions and their MIME types for validation
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
UPLOADS_DIR = Path("uploads")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent path traversal attacks.

    Operations:
      - Extract just the basename (strip any directory components)
      - Replace any non-alphanumeric characters (except . - _) with underscores
      - Lowercase the result for consistency

    Args:
        filename: Raw filename from the HTTP upload request.

    Returns:
        A safe, normalized filename string.
    """
    # Strip directory components — prevents path traversal: "../../etc/passwd"
    basename = Path(filename).name
    # Keep only safe characters
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", basename)
    return safe


def validate_extension(filename: str) -> Optional[str]:
    """
    Validate that the uploaded file has a supported extension.

    Args:
        filename: Sanitized filename.

    Returns:
        Error message string if invalid, None if valid.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return None


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """
    Write uploaded file bytes to the uploads/ directory.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Sanitized filename to save as.

    Returns:
        Path to the saved file.

    Raises:
        OSError: If the file cannot be written.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / filename

    dest.write_bytes(file_bytes)
    logger.debug(f"File saved | path={dest} | size={len(file_bytes)} bytes")
    return dest


def cleanup_file(file_path: Path) -> None:
    """
    Delete a temporary uploaded file after processing is complete.

    Silently ignores errors — a missing file on cleanup is not fatal.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Temp file cleaned up | path={file_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temp file | path={file_path} | error={e}")
