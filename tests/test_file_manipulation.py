"""
Unit tests for file manipulation service.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import os

from src.file_access.manipulator import FileManipulator, FileOperation
from src.file_access.transaction import TransactionManager


class TestFileManipulator:
    """Test FileManipulator functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as target_dir:
                yield source_dir, target_dir

    @pytest.fixture
    def sample_files(self, temp_dirs):
        """Create sample files for testing."""
        source_dir, _ = temp_dirs

        # Create test files
        files = []
        for i in range(3):
            file_path = Path(source_dir) / f"test_file_{i}.txt"
            file_path.write_text(f"Test content {i}")
            files.append(file_path)

        # Create subdirectory with file
        subdir = Path(source_dir) / "subdir"
        subdir.mkdir()
        sub_file = subdir / "sub_file.txt"
        sub_file.write_text("Subdirectory file")
        files.append(sub_file)

        return files

    def test_initialization(self, temp_dirs):
        """Test FileManipulator initialization."""
        _, target_dir = temp_dirs

        manipulator = FileManipulator(target_dir, dry_run=False)
        assert manipulator.base_directory == Path(target_dir)
        assert not manipulator.dry_run
        assert manipulator.verify_integrity
        assert manipulator.conflict_strategy == "rename"

    def test_move_file(self, temp_dirs, sample_files):
        """Test moving a file."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, dry_run=False)

        source_file = sample_files[0]
        target_path = "organized/documents/test.txt"

        # Move file
        success = manipulator.organize_file(
            str(source_file), target_path, operation="move"
        )

        assert success
        assert not source_file.exists()  # Original should be gone
        assert (Path(target_dir) / target_path).exists()  # Target should exist

        # Verify content
        target_content = (Path(target_dir) / target_path).read_text()
        assert target_content == "Test content 0"

    def test_copy_file(self, temp_dirs, sample_files):
        """Test copying a file."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, dry_run=False)

        source_file = sample_files[0]
        target_path = "organized/documents/test.txt"

        # Copy file
        success = manipulator.organize_file(
            str(source_file), target_path, operation="copy"
        )

        assert success
        assert source_file.exists()  # Original should still exist
        assert (Path(target_dir) / target_path).exists()  # Target should exist

        # Verify content
        source_content = source_file.read_text()
        target_content = (Path(target_dir) / target_path).read_text()
        assert source_content == target_content

    def test_dry_run_mode(self, temp_dirs, sample_files):
        """Test dry run mode."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, dry_run=True)

        source_file = sample_files[0]
        target_path = "organized/documents/test.txt"

        # Try to move file in dry run mode
        success = manipulator.organize_file(
            str(source_file), target_path, operation="move"
        )

        assert success
        assert source_file.exists()  # Original should still exist
        assert not (Path(target_dir) / target_path).exists()  # Target should not exist

        # Check that operation was logged
        assert len(manipulator.operations_log) == 1
        assert manipulator.operations_log[0].success

    def test_conflict_resolution_rename(self, temp_dirs, sample_files):
        """Test conflict resolution with rename strategy."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, conflict_strategy="rename")

        source_file1 = sample_files[0]
        source_file2 = sample_files[1]
        target_path = "organized/test.txt"

        # Move first file
        success1 = manipulator.organize_file(
            str(source_file1), target_path, operation="move"
        )
        assert success1
        assert (Path(target_dir) / target_path).exists()

        # Move second file with same target
        success2 = manipulator.organize_file(
            str(source_file2), target_path, operation="move"
        )
        assert success2

        # Check that second file was renamed
        renamed_path = Path(target_dir) / "organized" / "test_1.txt"
        assert renamed_path.exists()

    def test_conflict_resolution_skip(self, temp_dirs, sample_files):
        """Test conflict resolution with skip strategy."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, conflict_strategy="skip")

        source_file1 = sample_files[0]
        source_file2 = sample_files[1]
        target_path = "organized/test.txt"

        # Move first file
        success1 = manipulator.organize_file(
            str(source_file1), target_path, operation="move"
        )
        assert success1

        # Try to move second file with same target
        success2 = manipulator.organize_file(
            str(source_file2), target_path, operation="move"
        )
        assert not success2  # Should fail/skip
        assert source_file2.exists()  # Original should still exist

    def test_integrity_check(self, temp_dirs, sample_files):
        """Test file integrity verification."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir, verify_integrity=True)

        source_file = sample_files[0]
        target_path = "organized/test.txt"

        # Calculate checksum before
        import hashlib

        original_content = source_file.read_bytes()
        original_checksum = hashlib.sha256(original_content).hexdigest()

        # Move file
        success = manipulator.organize_file(
            str(source_file), target_path, operation="move"
        )
        assert success

        # Verify checksum after
        target_file = Path(target_dir) / target_path
        target_content = target_file.read_bytes()
        target_checksum = hashlib.sha256(target_content).hexdigest()

        assert original_checksum == target_checksum

    def test_batch_organize(self, temp_dirs, sample_files):
        """Test batch organization."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir)

        # Prepare batch operations
        files_with_paths = [
            (str(sample_files[0]), "documents/file1.txt"),
            (str(sample_files[1]), "documents/file2.txt"),
            (str(sample_files[2]), "images/file3.txt"),
        ]

        # Batch organize
        results = manipulator.batch_organize(files_with_paths)

        assert results["total"] == 3
        assert results["successful"] == 3
        assert results["failed"] == 0

        # Verify files are organized
        assert (Path(target_dir) / "documents" / "file1.txt").exists()
        assert (Path(target_dir) / "documents" / "file2.txt").exists()
        assert (Path(target_dir) / "images" / "file3.txt").exists()

    def test_rollback_operations(self, temp_dirs, sample_files):
        """Test rollback functionality."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir)

        source_file = sample_files[0]
        original_content = source_file.read_text()
        target_path = "organized/test.txt"

        # Move file
        success = manipulator.organize_file(
            str(source_file), target_path, operation="move"
        )
        assert success
        assert not source_file.exists()

        # Rollback
        rolled_back = manipulator.rollback_operations()
        assert rolled_back == 1

        # Verify file is back
        assert source_file.exists()
        assert source_file.read_text() == original_content
        assert not (Path(target_dir) / target_path).exists()

    def test_operation_logging(self, temp_dirs, sample_files):
        """Test operation logging."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir)

        # Perform some operations
        manipulator.organize_file(str(sample_files[0]), "test1.txt", operation="move")
        manipulator.organize_file(str(sample_files[1]), "test2.txt", operation="copy")

        # Check logs
        assert len(manipulator.operations_log) == 2
        assert manipulator.operations_log[0].operation_type == "move"
        assert manipulator.operations_log[1].operation_type == "copy"

        # Export logs
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            manipulator.export_operations_log(f.name)

        # Verify export
        with open(f.name, "r") as f:
            exported_logs = json.load(f)

        assert len(exported_logs) == 2
        assert exported_logs[0]["operation_type"] == "move"

    def test_verification(self, temp_dirs, sample_files):
        """Test organization verification."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir)

        # Organize files
        manipulator.organize_file(str(sample_files[0]), "docs/file1.txt")
        manipulator.organize_file(str(sample_files[1]), "docs/file2.txt")

        # Define expected structure
        expected_structure = {"docs": ["file1.txt", "file2.txt"]}

        # Verify
        verification = manipulator.verify_organization(expected_structure)

        assert verification["verified"]
        assert len(verification["missing_files"]) == 0
        assert len(verification["unexpected_files"]) == 0

    def test_report_generation(self, temp_dirs, sample_files):
        """Test report generation."""
        _, target_dir = temp_dirs
        manipulator = FileManipulator(target_dir)

        # Perform operations
        manipulator.organize_file(str(sample_files[0]), "success.txt")
        manipulator.organize_file("/nonexistent/file.txt", "fail.txt")

        # Generate report
        report = manipulator.create_organization_report()

        assert "File Organization Report" in report
        assert "Total Operations: 2" in report
        assert "Successful: 1" in report
        assert "Failed: 1" in report


class TestTransactionManager:
    """Test TransactionManager functionality."""

    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as target_dir:
                with tempfile.TemporaryDirectory() as log_dir:
                    # Create test files
                    files = []
                    for i in range(3):
                        file_path = Path(source_dir) / f"file_{i}.txt"
                        file_path.write_text(f"Content {i}")
                        files.append(file_path)

                    # Create managers
                    manipulator = FileManipulator(target_dir)
                    transaction_manager = TransactionManager(manipulator, log_dir)

                    yield {
                        "manipulator": manipulator,
                        "transaction_manager": transaction_manager,
                        "files": files,
                        "target_dir": target_dir,
                        "log_dir": log_dir,
                    }

    def test_transaction_lifecycle(self, setup):
        """Test basic transaction lifecycle."""
        tm = setup["transaction_manager"]
        files = setup["files"]

        # Begin transaction
        txn_id = tm.begin_transaction({"test": True})
        assert txn_id is not None
        assert tm.current_transaction is not None
        assert tm.current_transaction.status == "pending"

        # Add operations
        success1 = tm.add_operation(str(files[0]), "docs/file1.txt")
        success2 = tm.add_operation(str(files[1]), "docs/file2.txt")

        assert success1
        assert success2
        assert len(tm.current_transaction.operations) == 2

        # Commit transaction
        commit_success = tm.commit_transaction()
        assert commit_success
        assert tm.current_transaction is None

        # Check transaction in history
        assert len(tm.transaction_history) == 1
        assert tm.transaction_history[0].status == "committed"

    def test_transaction_rollback(self, setup):
        """Test transaction rollback."""
        tm = setup["transaction_manager"]
        files = setup["files"]

        # Begin transaction
        txn_id = tm.begin_transaction()

        # Add operations
        tm.add_operation(str(files[0]), "docs/file1.txt", operation_type="move")
        tm.add_operation(str(files[1]), "docs/file2.txt", operation_type="move")

        # Verify files were moved
        assert not files[0].exists()
        assert not files[1].exists()

        # Rollback transaction
        rollback_success = tm.rollback_transaction()
        assert rollback_success

        # Verify files are back
        assert files[0].exists()
        assert files[1].exists()
        assert files[0].read_text() == "Content 0"
        assert files[1].read_text() == "Content 1"

    def test_transaction_status(self, setup):
        """Test transaction status retrieval."""
        tm = setup["transaction_manager"]
        files = setup["files"]

        # Begin transaction
        txn_id = tm.begin_transaction()

        # Check status
        status = tm.get_transaction_status(txn_id)
        assert status is not None
        assert status["transaction_id"] == txn_id
        assert status["status"] == "pending"

        # Add operation and commit
        tm.add_operation(str(files[0]), "test.txt")
        tm.commit_transaction()

        # Check status again
        status = tm.get_transaction_status(txn_id)
        assert status["status"] == "committed"

    def test_transaction_recovery(self, setup):
        """Test transaction recovery."""
        tm = setup["transaction_manager"]
        files = setup["files"]

        # Begin transaction
        txn_id = tm.begin_transaction()

        # Add operations
        tm.add_operation(str(files[0]), "docs/file1.txt", operation_type="move")

        # Simulate crash by clearing current transaction
        tm.current_transaction = None

        # Recover transaction
        recovery_success = tm.recover_transaction(txn_id)
        assert recovery_success

        # Verify file was rolled back
        assert files[0].exists()

    def test_transaction_listing(self, setup):
        """Test listing transactions."""
        tm = setup["transaction_manager"]
        files = setup["files"]

        # Create multiple transactions
        for i in range(3):
            txn_id = tm.begin_transaction()
            tm.add_operation(str(files[i]), f"test{i}.txt")
            tm.commit_transaction()

        # List all transactions
        transactions = tm.list_transactions()
        assert len(transactions) >= 3

        # List by status
        committed = tm.list_transactions(status="committed")
        assert all(t["status"] == "committed" for t in committed)
