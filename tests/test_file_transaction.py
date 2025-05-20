"""
Unit tests for the file transaction module.
"""

import pytest
import shutil
from pathlib import Path
import tempfile
import os

from src.utils.file_transaction import FileTransaction, OperationType, Operation


class TestFileTransaction:
    """Test the FileTransaction class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.test_dir = self.temp_dir / "test_files"
        self.test_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_mkdir_operation(self):
        """Test directory creation operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        new_dir = self.test_dir / "new_directory"
        transaction.add_mkdir(new_dir)

        # Execute transaction
        assert transaction.execute()
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_copy_operation(self):
        """Test file copy operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create source file
        source_file = self.test_dir / "source.txt"
        source_file.write_text("test content")

        dest_file = self.test_dir / "dest.txt"
        transaction.add_copy(source_file, dest_file)

        # Execute transaction
        assert transaction.execute()
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"
        assert source_file.exists()  # Original should still exist

    def test_move_operation(self):
        """Test file move operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create source file
        source_file = self.test_dir / "source.txt"
        source_file.write_text("test content")

        dest_file = self.test_dir / "moved.txt"
        transaction.add_move(source_file, dest_file)

        # Execute transaction
        assert transaction.execute()
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"
        assert not source_file.exists()  # Original should be gone

    def test_delete_operation(self):
        """Test file deletion operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create file to delete
        file_to_delete = self.test_dir / "delete_me.txt"
        file_to_delete.write_text("delete this")

        transaction.add_delete(file_to_delete)

        # Execute transaction
        assert transaction.execute()
        assert not file_to_delete.exists()

    def test_write_operation(self):
        """Test file write operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        file_path = self.test_dir / "write_test.txt"
        transaction.add_write(file_path, "new content")

        # Execute transaction
        assert transaction.execute()
        assert file_path.exists()
        assert file_path.read_text() == "new content"

    def test_rollback_mkdir(self):
        """Test rollback of directory creation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        new_dir = self.test_dir / "rollback_dir"
        transaction.add_mkdir(new_dir)

        # Create a failing operation
        transaction.add_move(
            self.test_dir / "non_existent.txt", self.test_dir / "dest.txt"
        )

        # Execute should fail and rollback
        with pytest.raises(Exception):
            transaction.execute()

        # Directory should be removed if it was created and is empty
        assert not new_dir.exists()

    def test_rollback_copy(self):
        """Test rollback of copy operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create source file
        source_file = self.test_dir / "source.txt"
        source_file.write_text("original content")

        # Create destination file that will be overwritten
        dest_file = self.test_dir / "dest.txt"
        dest_file.write_text("existing content")

        transaction.add_copy(source_file, dest_file)

        # Add a failing operation
        transaction.add_move(
            self.test_dir / "non_existent.txt", self.test_dir / "moved.txt"
        )

        # Execute should fail and rollback
        with pytest.raises(Exception):
            transaction.execute()

        # Destination should have original content
        assert dest_file.exists()
        assert dest_file.read_text() == "existing content"

    def test_rollback_move(self):
        """Test rollback of move operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create source file
        source_file = self.test_dir / "source.txt"
        source_file.write_text("test content")

        dest_file = self.test_dir / "moved.txt"

        transaction.add_move(source_file, dest_file)

        # Add a failing operation
        transaction.add_delete(self.test_dir / "non_existent.txt")

        # Execute should fail and rollback
        with pytest.raises(Exception):
            transaction.execute()

        # File should be back at original location
        assert source_file.exists()
        assert source_file.read_text() == "test content"
        assert not dest_file.exists()

    def test_rollback_delete(self):
        """Test rollback of delete operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create file to delete
        file_to_delete = self.test_dir / "delete_me.txt"
        file_to_delete.write_text("important data")

        transaction.add_delete(file_to_delete)

        # Add a failing operation
        transaction.add_move(
            self.test_dir / "non_existent.txt", self.test_dir / "moved.txt"
        )

        # Execute should fail and rollback
        with pytest.raises(Exception):
            transaction.execute()

        # File should be restored
        assert file_to_delete.exists()
        assert file_to_delete.read_text() == "important data"

    def test_rollback_write(self):
        """Test rollback of write operation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create existing file
        file_path = self.test_dir / "write_test.txt"
        file_path.write_text("original content")

        transaction.add_write(file_path, "new content")

        # Add a failing operation
        transaction.add_delete(self.test_dir / "non_existent.txt")

        # Execute should fail and rollback
        with pytest.raises(Exception):
            transaction.execute()

        # File should have original content
        assert file_path.exists()
        assert file_path.read_text() == "original content"

    def test_multiple_operations(self):
        """Test transaction with multiple operations."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create test files
        file1 = self.test_dir / "file1.txt"
        file1.write_text("content 1")

        file2 = self.test_dir / "file2.txt"
        file2.write_text("content 2")

        # Add multiple operations
        new_dir = self.test_dir / "new_folder"
        transaction.add_mkdir(new_dir)

        transaction.add_copy(file1, new_dir / "copied.txt")
        transaction.add_move(file2, new_dir / "moved.txt")

        new_file = new_dir / "created.txt"
        transaction.add_write(new_file, "created content")

        # Execute transaction
        assert transaction.execute()

        # Verify all operations
        assert new_dir.exists()
        assert (new_dir / "copied.txt").exists()
        assert (new_dir / "copied.txt").read_text() == "content 1"
        assert (new_dir / "moved.txt").exists()
        assert (new_dir / "moved.txt").read_text() == "content 2"
        assert not file2.exists()
        assert new_file.exists()
        assert new_file.read_text() == "created content"

    def test_committed_transaction(self):
        """Test that committed transactions cannot be executed again."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        new_dir = self.test_dir / "test_dir"
        transaction.add_mkdir(new_dir)

        # Execute and commit
        assert transaction.execute()
        assert transaction.committed

        # Try to execute again
        with pytest.raises(RuntimeError, match="Transaction already committed"):
            transaction.execute()

    def test_transaction_log(self):
        """Test transaction log generation."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Create test file
        source_file = self.test_dir / "source.txt"
        source_file.write_text("content")

        dest_file = self.test_dir / "dest.txt"

        # Add operations
        transaction.add_copy(source_file, dest_file)
        transaction.add_delete(source_file)

        # Execute
        transaction.execute()

        # Get transaction log
        log = transaction.get_transaction_log()

        assert log["committed"] is True
        assert len(log["operations"]) == 2
        assert log["operations"][0]["type"] == "copy"
        assert log["operations"][0]["executed"] is True
        assert log["operations"][1]["type"] == "delete"
        assert log["operations"][1]["executed"] is True

    def test_save_transaction_log(self):
        """Test saving transaction log to file."""
        transaction = FileTransaction(backup_dir=self.backup_dir)

        # Add operation
        new_dir = self.test_dir / "test_dir"
        transaction.add_mkdir(new_dir)

        # Execute
        transaction.execute()

        # Save log
        log_file = self.test_dir / "transaction.log"
        transaction.save_transaction_log(log_file)

        # Verify log file
        assert log_file.exists()

        import json

        with open(log_file) as f:
            saved_log = json.load(f)

        assert saved_log["committed"] is True
        assert len(saved_log["operations"]) == 1
        assert saved_log["operations"][0]["type"] == "mkdir"


if __name__ == "__main__":
    pytest.main([__file__])
