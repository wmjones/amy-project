import os
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Data class to hold file information."""

    path: str
    name: str
    extension: str
    size: int
    created_time: datetime
    modified_time: datetime
    mime_type: str
    is_supported: bool


class FileSystemAccessor:
    """Handles local file system access and file scanning."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".bmp",  # Images
        ".pdf",  # PDFs
        ".docx",
        ".doc",  # Word documents
        ".txt",
        ".text",  # Text files
    }

    def __init__(self, root_directory: str):
        """Initialize the file system accessor.

        Args:
            root_directory: The root directory to scan
        """
        self.root_directory = Path(root_directory)
        if not self.root_directory.exists():
            raise ValueError(f"Directory does not exist: {root_directory}")
        if not self.root_directory.is_dir():
            raise ValueError(f"Path is not a directory: {root_directory}")

        self.logger = logging.getLogger(__name__)

    def scan_directory(self, recursive: bool = True) -> List[FileInfo]:
        """Scan the directory and return list of files.

        Args:
            recursive: Whether to scan subdirectories

        Returns:
            List of FileInfo objects
        """
        file_list = []

        if recursive:
            # Use glob for recursive scanning
            pattern = "**/*"
        else:
            pattern = "*"

        self.logger.info(f"Scanning directory: {self.root_directory}")

        for file_path in self.root_directory.glob(pattern):
            if file_path.is_file():
                try:
                    file_info = self._create_file_object(file_path)
                    file_list.append(file_info)
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")

        self.logger.info(f"Found {len(file_list)} files")
        return file_list

    def get_supported_files(self, recursive: bool = True) -> List[FileInfo]:
        """Get only supported files from the directory.

        Args:
            recursive: Whether to scan subdirectories

        Returns:
            List of supported FileInfo objects
        """
        all_files = self.scan_directory(recursive)
        supported_files = [f for f in all_files if f.is_supported]

        self.logger.info(
            f"Found {len(supported_files)} supported files out of {len(all_files)} total"
        )
        return supported_files

    def _create_file_object(self, file_path: Path) -> FileInfo:
        """Create a FileInfo object from a file path.

        Args:
            file_path: Path to the file

        Returns:
            FileInfo object with file metadata
        """
        stat = file_path.stat()
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return FileInfo(
            path=str(file_path),
            name=file_path.name,
            extension=extension,
            size=stat.st_size,
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            mime_type=mime_type or "unknown",
            is_supported=self._is_supported_file(extension),
        )

    def _is_supported_file(self, extension: str) -> bool:
        """Check if a file extension is supported.

        Args:
            extension: File extension (with dot)

        Returns:
            True if supported, False otherwise
        """
        return extension.lower() in self.SUPPORTED_EXTENSIONS

    def get_directory_stats(self) -> Dict[str, int]:
        """Get statistics about the directory.

        Returns:
            Dictionary with file counts by type
        """
        files = self.scan_directory()
        stats = {
            "total_files": len(files),
            "supported_files": sum(1 for f in files if f.is_supported),
            "total_size": sum(f.size for f in files),
            "by_extension": {},
        }

        # Count by extension
        for file in files:
            ext = file.extension.lower()
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

        return stats

    def filter_by_extension(self, extensions: List[str]) -> List[FileInfo]:
        """Get files with specific extensions.

        Args:
            extensions: List of extensions to filter by

        Returns:
            List of FileInfo objects matching the extensions
        """
        extensions_lower = {ext.lower() for ext in extensions}
        files = self.scan_directory()
        return [f for f in files if f.extension.lower() in extensions_lower]

    def filter_by_size(
        self, min_size: Optional[int] = None, max_size: Optional[int] = None
    ) -> List[FileInfo]:
        """Get files within a size range.

        Args:
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes

        Returns:
            List of FileInfo objects within size range
        """
        files = self.scan_directory()
        filtered = []

        for file in files:
            if min_size is not None and file.size < min_size:
                continue
            if max_size is not None and file.size > max_size:
                continue
            filtered.append(file)

        return filtered
