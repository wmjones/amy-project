"""
Utility functions for file operations.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
import magic
import logging

logger = logging.getLogger(__name__)


def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex string of the file hash
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def get_file_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type of a file using python-magic.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string or None if detection fails
    """
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        logger.warning(f"Failed to detect MIME type for {file_path}: {e}")
        return None


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Convert a string to a safe filename.

    Args:
        filename: Original filename
        max_length: Maximum length of filename

    Returns:
        Safe filename string
    """
    # Remove invalid characters
    invalid_chars = '<>:"|?*\\/\0'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    return filename


def ensure_directory(directory_path: str) -> Path:
    """Ensure a directory exists, create if necessary.

    Args:
        directory_path: Path to directory

    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def is_binary_file(file_path: str) -> bool:
    """Check if a file is binary or text.

    Args:
        file_path: Path to the file

    Returns:
        True if binary, False if text
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(512)

        # Check for null bytes
        if b"\0" in chunk:
            return True

        # Try to decode as UTF-8
        try:
            chunk.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True

    except Exception as e:
        logger.warning(f"Error checking if file is binary: {e}")
        return True  # Assume binary if can't determine
