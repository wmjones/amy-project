import os
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from src.file_access.local_accessor import FileSystemAccessor, FileInfo


class TestFileSystemAccessor:
    """Test FileSystemAccessor functionality."""

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "test_image.jpg": b"fake jpeg data",
                "test_document.pdf": b"fake pdf data",
                "test_text.txt": b"fake text data",
                "unsupported.xyz": b"fake unsupported data",
                "subdir/nested_image.png": b"fake png data",
            }

            for file_path, content in test_files.items():
                full_path = Path(temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_bytes(content)

            yield temp_dir

    def test_initialization_valid_directory(self, temp_directory):
        """Test initialization with valid directory."""
        accessor = FileSystemAccessor(temp_directory)
        assert accessor.root_directory == Path(temp_directory)

    def test_initialization_invalid_directory(self):
        """Test initialization with invalid directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            FileSystemAccessor("/nonexistent/directory")

    def test_initialization_not_directory(self, temp_directory):
        """Test initialization with file instead of directory."""
        file_path = Path(temp_directory) / "test_file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="Path is not a directory"):
            FileSystemAccessor(str(file_path))

    def test_scan_directory_recursive(self, temp_directory):
        """Test recursive directory scanning."""
        accessor = FileSystemAccessor(temp_directory)
        files = accessor.scan_directory(recursive=True)

        assert len(files) == 5
        file_names = {f.name for f in files}
        assert "test_image.jpg" in file_names
        assert "nested_image.png" in file_names

    def test_scan_directory_non_recursive(self, temp_directory):
        """Test non-recursive directory scanning."""
        accessor = FileSystemAccessor(temp_directory)
        files = accessor.scan_directory(recursive=False)

        assert len(files) == 4  # Only files in root, not in subdir
        file_names = {f.name for f in files}
        assert "test_image.jpg" in file_names
        assert "nested_image.png" not in file_names

    def test_get_supported_files(self, temp_directory):
        """Test getting only supported files."""
        accessor = FileSystemAccessor(temp_directory)
        supported_files = accessor.get_supported_files()

        assert len(supported_files) == 4  # All except .xyz file
        file_names = {f.name for f in supported_files}
        assert "unsupported.xyz" not in file_names
        assert "test_image.jpg" in file_names

    def test_file_info_creation(self, temp_directory):
        """Test FileInfo object creation."""
        accessor = FileSystemAccessor(temp_directory)
        files = accessor.scan_directory()

        jpg_file = next(f for f in files if f.name == "test_image.jpg")

        assert jpg_file.extension == ".jpg"
        assert jpg_file.size > 0
        assert jpg_file.is_supported
        assert isinstance(jpg_file.created_time, datetime)
        assert isinstance(jpg_file.modified_time, datetime)

    def test_supported_extensions(self):
        """Test supported file extension checking."""
        accessor = FileSystemAccessor(".")  # Current directory

        assert accessor._is_supported_file(".jpg")
        assert accessor._is_supported_file(".JPEG")  # Case insensitive
        assert accessor._is_supported_file(".pdf")
        assert accessor._is_supported_file(".docx")
        assert not accessor._is_supported_file(".xyz")
        assert not accessor._is_supported_file(".exe")

    def test_get_directory_stats(self, temp_directory):
        """Test directory statistics."""
        accessor = FileSystemAccessor(temp_directory)
        stats = accessor.get_directory_stats()

        assert stats["total_files"] == 5
        assert stats["supported_files"] == 4
        assert stats["total_size"] > 0
        assert ".jpg" in stats["by_extension"]
        assert ".xyz" in stats["by_extension"]

    def test_filter_by_extension(self, temp_directory):
        """Test filtering files by extension."""
        accessor = FileSystemAccessor(temp_directory)

        # Filter for images
        images = accessor.filter_by_extension([".jpg", ".png"])
        assert len(images) == 2

        # Filter for documents
        docs = accessor.filter_by_extension([".pdf", ".docx"])
        assert len(docs) == 1

    def test_filter_by_size(self, temp_directory):
        """Test filtering files by size."""
        accessor = FileSystemAccessor(temp_directory)

        # All files in our test are small
        small_files = accessor.filter_by_size(max_size=1000)
        assert len(small_files) == 5

        # No files should be larger than 1MB
        large_files = accessor.filter_by_size(min_size=1_000_000)
        assert len(large_files) == 0

    def test_empty_directory(self):
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            accessor = FileSystemAccessor(temp_dir)
            files = accessor.scan_directory()

            assert len(files) == 0
            assert accessor.get_directory_stats()["total_files"] == 0
