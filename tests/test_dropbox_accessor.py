"""
Unit tests for the Dropbox accessor module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from dropbox.exceptions import ApiError, AuthError

from src.file_access.dropbox_accessor import DropboxAccessor, DropboxFile


class TestDropboxAccessor:
    """Test the DropboxAccessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app_key = "test_app_key"
        self.app_secret = "test_app_secret"
        self.refresh_token = "test_refresh_token"

    @patch("dropbox.Dropbox")
    def test_authenticate_with_refresh_token(self, mock_dropbox):
        """Test authentication with refresh token."""
        # Mock successful authentication
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        # Verify Dropbox client was created with correct parameters
        mock_dropbox.assert_called_once_with(
            app_key=self.app_key,
            app_secret=self.app_secret,
            oauth2_refresh_token=self.refresh_token,
        )

        # Verify connection was tested
        mock_client.users_get_current_account.assert_called_once()

    @patch("dropbox.Dropbox")
    def test_authenticate_with_access_token(self, mock_dropbox):
        """Test authentication with access token."""
        access_token = "test_access_token"
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, access_token=access_token
        )

        # Verify Dropbox client was created with access token
        mock_dropbox.assert_called_once_with(access_token)
        mock_client.users_get_current_account.assert_called_once()

    @patch("dropbox.Dropbox")
    def test_list_files(self, mock_dropbox):
        """Test listing files from Dropbox."""
        # Mock Dropbox client
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Create mock file metadata
        mock_file1 = Mock(spec=FileMetadata)
        mock_file1.id = "id1"
        mock_file1.name = "test1.pdf"
        mock_file1.path_display = "/test1.pdf"
        mock_file1.size = 1024
        mock_file1.server_modified = datetime.now()
        mock_file1.content_hash = "hash1"
        mock_file1.is_downloadable = True

        mock_file2 = Mock(spec=FileMetadata)
        mock_file2.id = "id2"
        mock_file2.name = "test2.jpg"
        mock_file2.path_display = "/folder/test2.jpg"
        mock_file2.size = 2048
        mock_file2.server_modified = datetime.now()
        mock_file2.content_hash = "hash2"
        mock_file2.is_downloadable = True

        # Mock folder metadata
        mock_folder = Mock(spec=FolderMetadata)
        mock_folder.name = "folder"

        # Mock list folder result
        mock_result = Mock()
        mock_result.entries = [mock_file1, mock_folder, mock_file2]
        mock_result.has_more = False

        mock_client.files_list_folder.return_value = mock_result

        # Create accessor and list files
        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        files = accessor.list_files("/", recursive=True)

        # Verify results
        assert len(files) == 2  # Should only return files, not folders
        assert files[0].name == "test1.pdf"
        assert files[0].path == "/test1.pdf"
        assert files[1].name == "test2.jpg"
        assert files[1].path == "/folder/test2.jpg"

    @patch("dropbox.Dropbox")
    def test_list_files_with_filter(self, mock_dropbox):
        """Test listing files with type filter."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Create mock files with different extensions
        mock_pdf = Mock(spec=FileMetadata)
        mock_pdf.name = "document.pdf"
        mock_pdf.path_display = "/document.pdf"
        # Set other required attributes
        mock_pdf.id = "id1"
        mock_pdf.size = 1024
        mock_pdf.server_modified = datetime.now()
        mock_pdf.content_hash = "hash1"
        mock_pdf.is_downloadable = True

        mock_jpg = Mock(spec=FileMetadata)
        mock_jpg.name = "image.jpg"
        mock_jpg.path_display = "/image.jpg"
        # Set other required attributes
        mock_jpg.id = "id2"
        mock_jpg.size = 2048
        mock_jpg.server_modified = datetime.now()
        mock_jpg.content_hash = "hash2"
        mock_jpg.is_downloadable = True

        mock_result = Mock()
        mock_result.entries = [mock_pdf, mock_jpg]
        mock_result.has_more = False

        mock_client.files_list_folder.return_value = mock_result

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        # Filter for PDFs only
        files = accessor.list_files("/", file_types=[".pdf"])

        assert len(files) == 1
        assert files[0].name == "document.pdf"

    @patch("dropbox.Dropbox")
    def test_download_file(self, mock_dropbox):
        """Test downloading a file."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Mock file metadata
        mock_metadata = Mock(spec=FileMetadata)
        mock_metadata.size = 1024

        mock_client.files_get_metadata.return_value = mock_metadata

        # For small files, the accessor uses files_download_to_file
        mock_client.files_download_to_file.return_value = None

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        # Test download
        local_path = Path("/tmp/test.pdf")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        result = accessor.download_file("/test.pdf", local_path, show_progress=False)

        assert result == local_path
        mock_client.files_download_to_file.assert_called_once_with(
            str(local_path), "/test.pdf"
        )

    @patch("dropbox.Dropbox")
    def test_create_folder(self, mock_dropbox):
        """Test creating a folder."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        result = accessor.create_folder("/new_folder")

        assert result is True
        mock_client.files_create_folder_v2.assert_called_once_with("/new_folder")

    @patch("dropbox.Dropbox")
    def test_create_folder_already_exists(self, mock_dropbox):
        """Test creating a folder that already exists."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Mock folder exists error
        mock_error = Mock()
        mock_error.error.is_path.return_value = True
        mock_error.error.get_path.return_value.is_conflict.return_value = True

        mock_client.files_create_folder_v2.side_effect = ApiError(
            request_id="123",
            error=mock_error,
            user_message_text="Folder already exists",
            user_message_locale=None,
        )

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        result = accessor.create_folder("/existing_folder")

        assert result is True

    @patch("dropbox.Dropbox")
    def test_move_file(self, mock_dropbox):
        """Test moving a file."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        result = accessor.move_file("/old_path.pdf", "/new_path.pdf")

        assert result is True
        mock_client.files_move_v2.assert_called_once_with(
            "/old_path.pdf", "/new_path.pdf", autorename=True
        )

    @patch("dropbox.Dropbox")
    def test_copy_file(self, mock_dropbox):
        """Test copying a file."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        result = accessor.copy_file("/source.pdf", "/dest.pdf")

        assert result is True
        mock_client.files_copy_v2.assert_called_once_with(
            "/source.pdf", "/dest.pdf", autorename=True
        )

    @patch("dropbox.Dropbox")
    def test_delete_file(self, mock_dropbox):
        """Test deleting a file."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        result = accessor.delete_file("/file_to_delete.pdf")

        assert result is True
        mock_client.files_delete_v2.assert_called_once_with("/file_to_delete.pdf")

    @patch("dropbox.Dropbox")
    def test_search_files(self, mock_dropbox):
        """Test searching for files."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Mock search results
        mock_match1 = Mock()
        mock_match1.metadata = Mock(spec=FileMetadata)
        mock_match1.metadata.name = "invoice_2023.pdf"
        mock_match1.metadata.id = "id1"
        mock_match1.metadata.path_display = "/invoices/invoice_2023.pdf"
        mock_match1.metadata.size = 1024
        mock_match1.metadata.server_modified = datetime.now()
        mock_match1.metadata.content_hash = "hash1"
        mock_match1.metadata.is_downloadable = True

        mock_result = Mock()
        mock_result.matches = [mock_match1]

        mock_client.files_search.return_value = mock_result

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        files = accessor.search_files("invoice")

        assert len(files) == 1
        assert files[0].name == "invoice_2023.pdf"

    @patch("dropbox.Dropbox")
    def test_get_space_usage(self, mock_dropbox):
        """Test getting space usage information."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Mock space usage
        mock_space_usage = Mock()
        mock_space_usage.used = 1024 * 1024 * 1024  # 1GB
        mock_allocation = Mock()
        mock_allocation.allocated = 2 * 1024 * 1024 * 1024  # 2GB
        mock_space_usage.allocation.get_individual.return_value = mock_allocation

        mock_client.users_get_space_usage.return_value = mock_space_usage

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        usage = accessor.get_space_usage()

        assert usage["used"] == 1024 * 1024 * 1024
        assert usage["allocated"] == 2 * 1024 * 1024 * 1024

    @patch("dropbox.Dropbox")
    def test_error_handling(self, mock_dropbox):
        """Test error handling for API errors."""
        mock_client = Mock()
        mock_dropbox.return_value = mock_client

        # Mock API error
        mock_client.files_list_folder.side_effect = ApiError(
            request_id="123",
            error=None,
            user_message_text="API Error",
            user_message_locale=None,
        )

        accessor = DropboxAccessor(
            self.app_key, self.app_secret, refresh_token=self.refresh_token
        )

        with pytest.raises(ApiError):
            accessor.list_files("/")

    def test_create_file_object(self):
        """Test creating DropboxFile object from metadata."""
        # Create mock metadata
        mock_metadata = Mock(spec=FileMetadata)
        mock_metadata.id = "test_id"
        mock_metadata.name = "test.pdf"
        mock_metadata.path_display = "/folder/test.pdf"
        mock_metadata.size = 1024
        mock_metadata.server_modified = datetime.now()
        mock_metadata.content_hash = "test_hash"
        mock_metadata.is_downloadable = True

        # Create accessor with mocked client
        with patch("dropbox.Dropbox"):
            accessor = DropboxAccessor(
                self.app_key, self.app_secret, refresh_token=self.refresh_token
            )

        # Create file object
        file_obj = accessor._create_file_object(mock_metadata)

        assert file_obj.id == "test_id"
        assert file_obj.name == "test.pdf"
        assert file_obj.path == "/folder/test.pdf"
        assert file_obj.size == 1024
        assert file_obj.extension == ".pdf"


if __name__ == "__main__":
    pytest.main([__file__])
