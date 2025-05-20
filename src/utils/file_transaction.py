"""
Transaction-like file operations with rollback capability.
Ensures atomic file operations and provides recovery mechanisms.
"""

import os
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of file operations."""

    MKDIR = "mkdir"
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"
    WRITE = "write"
    CHMOD = "chmod"


class Operation:
    """Represents a single file operation."""

    def __init__(
        self,
        op_type: OperationType,
        source: Path,
        destination: Optional[Path] = None,
        data: Optional[Any] = None,
    ):
        self.op_type = op_type
        self.source = source
        self.destination = destination
        self.data = data
        self.executed = False
        self.backup_info: Optional[Dict[str, Any]] = None

    def __repr__(self):
        return f"Operation({self.op_type.value}, {self.source}, {self.destination})"


class FileTransaction:
    """Handle file operations in a transaction-like manner with rollback support."""

    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize file transaction.

        Args:
            backup_dir: Directory to store backups for rollback
        """
        self.operations: List[Operation] = []
        self.executed_operations: List[Operation] = []
        self.backup_dir = backup_dir or Path.home() / ".file_transaction_backups"
        self.transaction_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.committed = False
        self.logger = logging.getLogger("file_transaction")

        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def add_operation(
        self,
        op_type: OperationType,
        source: Path,
        destination: Optional[Path] = None,
        data: Optional[Any] = None,
    ):
        """Add an operation to the transaction."""
        operation = Operation(op_type, source, destination, data)
        self.operations.append(operation)
        self.logger.debug(f"Added operation: {operation}")

    def add_mkdir(self, path: Path):
        """Add directory creation operation."""
        self.add_operation(OperationType.MKDIR, path)

    def add_copy(self, source: Path, destination: Path):
        """Add file copy operation."""
        self.add_operation(OperationType.COPY, source, destination)

    def add_move(self, source: Path, destination: Path):
        """Add file move operation."""
        self.add_operation(OperationType.MOVE, source, destination)

    def add_delete(self, path: Path):
        """Add file deletion operation."""
        self.add_operation(OperationType.DELETE, path)

    def add_write(self, path: Path, data: str):
        """Add file write operation."""
        self.add_operation(OperationType.WRITE, path, data=data)

    def execute(self) -> bool:
        """
        Execute all operations in the transaction.

        Returns:
            True if all operations succeeded, False otherwise
        """
        if self.committed:
            raise RuntimeError("Transaction already committed")

        self.logger.info(
            f"Executing transaction {self.transaction_id} with {len(self.operations)} operations"
        )

        for operation in self.operations:
            try:
                self._execute_operation(operation)
                operation.executed = True
                self.executed_operations.append(operation)
            except Exception as e:
                self.logger.error(f"Operation failed: {operation}. Error: {str(e)}")
                self.rollback()
                raise e

        self.committed = True
        self.logger.info(f"Transaction {self.transaction_id} completed successfully")
        return True

    def _execute_operation(self, operation: Operation):
        """Execute a single operation."""
        self.logger.debug(f"Executing operation: {operation}")

        if operation.op_type == OperationType.MKDIR:
            # Backup: record if directory already exists
            operation.backup_info = {"existed": operation.source.exists()}
            operation.source.mkdir(parents=True, exist_ok=True)

        elif operation.op_type == OperationType.COPY:
            # Backup: record if destination exists
            if operation.destination.exists():
                backup_path = self._create_backup(operation.destination)
                operation.backup_info = {"backup_path": backup_path}
            shutil.copy2(operation.source, operation.destination)

        elif operation.op_type == OperationType.MOVE:
            # Backup: record original location
            operation.backup_info = {"original_path": operation.source}
            shutil.move(str(operation.source), str(operation.destination))

        elif operation.op_type == OperationType.DELETE:
            # Backup: create backup of file/directory
            backup_path = self._create_backup(operation.source)
            operation.backup_info = {"backup_path": backup_path}
            if operation.source.is_dir():
                shutil.rmtree(operation.source)
            else:
                operation.source.unlink()

        elif operation.op_type == OperationType.WRITE:
            # Backup: backup existing file if it exists
            if operation.source.exists():
                backup_path = self._create_backup(operation.source)
                operation.backup_info = {"backup_path": backup_path}
            with open(operation.source, "w") as f:
                f.write(operation.data)

        elif operation.op_type == OperationType.CHMOD:
            # Backup: record original permissions
            if operation.source.exists():
                operation.backup_info = {
                    "original_mode": oct(operation.source.stat().st_mode)
                }
            os.chmod(operation.source, operation.data)

    def _create_backup(self, path: Path) -> Path:
        """Create a backup of a file or directory."""
        backup_path = self.backup_dir / self.transaction_id / path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        if path.is_dir():
            shutil.copytree(path, backup_path)
        else:
            shutil.copy2(path, backup_path)

        return backup_path

    def rollback(self):
        """Rollback all executed operations."""
        if self.committed:
            self.logger.warning("Cannot rollback committed transaction")
            return

        self.logger.info(f"Rolling back transaction {self.transaction_id}")

        # Rollback in reverse order
        for operation in reversed(self.executed_operations):
            try:
                self._rollback_operation(operation)
            except Exception as e:
                self.logger.error(
                    f"Failed to rollback operation: {operation}. Error: {str(e)}"
                )
                # Continue with other rollbacks

        # Clean up backup directory for this transaction
        transaction_backup_dir = self.backup_dir / self.transaction_id
        if transaction_backup_dir.exists():
            shutil.rmtree(transaction_backup_dir)

    def _rollback_operation(self, operation: Operation):
        """Rollback a single operation."""
        self.logger.debug(f"Rolling back operation: {operation}")

        if operation.op_type == OperationType.MKDIR:
            # Only remove if we created it and it's empty
            if not operation.backup_info["existed"]:
                if operation.source.exists() and not any(operation.source.iterdir()):
                    operation.source.rmdir()

        elif operation.op_type == OperationType.COPY:
            # Remove the copy and restore backup if exists
            if operation.destination.exists():
                operation.destination.unlink()
            if operation.backup_info and "backup_path" in operation.backup_info:
                shutil.move(
                    str(operation.backup_info["backup_path"]),
                    str(operation.destination),
                )

        elif operation.op_type == OperationType.MOVE:
            # Move back to original location
            if operation.destination.exists():
                shutil.move(
                    str(operation.destination),
                    str(operation.backup_info["original_path"]),
                )

        elif operation.op_type == OperationType.DELETE:
            # Restore from backup
            if operation.backup_info and "backup_path" in operation.backup_info:
                backup_path = operation.backup_info["backup_path"]
                if backup_path.is_dir():
                    shutil.copytree(backup_path, operation.source)
                else:
                    shutil.copy2(backup_path, operation.source)

        elif operation.op_type == OperationType.WRITE:
            # Restore original file if it existed
            if operation.backup_info and "backup_path" in operation.backup_info:
                shutil.copy2(operation.backup_info["backup_path"], operation.source)
            else:
                # File didn't exist before, remove it
                if operation.source.exists():
                    operation.source.unlink()

        elif operation.op_type == OperationType.CHMOD:
            # Restore original permissions
            if operation.backup_info and "original_mode" in operation.backup_info:
                os.chmod(
                    operation.source, int(operation.backup_info["original_mode"], 8)
                )

    def get_transaction_log(self) -> Dict[str, Any]:
        """Get a log of all operations in the transaction."""
        return {
            "transaction_id": self.transaction_id,
            "committed": self.committed,
            "operations": [
                {
                    "type": op.op_type.value,
                    "source": str(op.source),
                    "destination": str(op.destination) if op.destination else None,
                    "executed": op.executed,
                    "backup_info": self._serialize_backup_info(op.backup_info),
                }
                for op in self.operations
            ],
        }

    def _serialize_backup_info(
        self, backup_info: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Convert backup info to JSON-serializable format."""
        if not backup_info:
            return None

        serialized = {}
        for key, value in backup_info.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def save_transaction_log(self, filepath: Path):
        """Save transaction log to file."""
        log = self.get_transaction_log()
        with open(filepath, "w") as f:
            json.dump(log, f, indent=2)
        self.logger.info(f"Transaction log saved to {filepath}")
