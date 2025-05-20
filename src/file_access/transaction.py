"""
Transaction management for file operations.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

from .manipulator import FileManipulator, FileOperation

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Represents a file organization transaction."""

    transaction_id: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "pending"  # pending, committed, rolled_back, failed
    operations: List[FileOperation] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.operations is None:
            self.operations = []
        if self.metadata is None:
            self.metadata = {}


class TransactionManager:
    """Manage atomic file organization transactions."""

    def __init__(
        self, manipulator: FileManipulator, transaction_log_dir: Optional[str] = None
    ):
        """Initialize transaction manager.

        Args:
            manipulator: FileManipulator instance
            transaction_log_dir: Directory to store transaction logs
        """
        self.manipulator = manipulator
        self.current_transaction = None
        self.transaction_history = []

        # Set up transaction log directory
        if transaction_log_dir:
            self.log_dir = Path(transaction_log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = Path(tempfile.gettempdir()) / "file_organizer_transactions"
            self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TransactionManager initialized with log dir: {self.log_dir}")

    def begin_transaction(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin a new transaction.

        Args:
            metadata: Optional metadata for the transaction

        Returns:
            Transaction ID
        """
        if self.current_transaction and self.current_transaction.status == "pending":
            raise RuntimeError("Another transaction is already in progress")

        # Generate transaction ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transaction_id = f"txn_{timestamp}"

        # Create new transaction
        self.current_transaction = Transaction(
            transaction_id=transaction_id,
            start_time=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        # Save initial transaction state
        self._save_transaction_log(self.current_transaction)

        logger.info(f"Started transaction: {transaction_id}")
        return transaction_id

    def add_operation(
        self, source_path: str, target_path: str, operation_type: str = "move"
    ) -> bool:
        """Add an operation to the current transaction.

        Args:
            source_path: Source file path
            target_path: Target file path
            operation_type: 'move' or 'copy'

        Returns:
            True if operation was successful
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        if self.current_transaction.status != "pending":
            raise RuntimeError("Transaction is not in pending state")

        # Perform the operation
        success = self.manipulator.organize_file(
            source_path, target_path, operation_type
        )

        # Get the operation from manipulator's log
        if self.manipulator.operations_log:
            operation = self.manipulator.operations_log[-1]
            self.current_transaction.operations.append(operation)

            # Update transaction log
            self._save_transaction_log(self.current_transaction)

        return success

    def commit_transaction(self) -> bool:
        """Commit the current transaction.

        Returns:
            True if commit successful
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        if self.current_transaction.status != "pending":
            raise RuntimeError("Transaction is not in pending state")

        try:
            # Check if all operations were successful
            failed_operations = [
                op for op in self.current_transaction.operations if not op.success
            ]

            if failed_operations:
                logger.warning(
                    f"Transaction {self.current_transaction.transaction_id} has failed operations"
                )
                self.current_transaction.status = "failed"
            else:
                self.current_transaction.status = "committed"

            self.current_transaction.end_time = datetime.now().isoformat()

            # Save final transaction state
            self._save_transaction_log(self.current_transaction)

            # Add to history
            self.transaction_history.append(self.current_transaction)

            logger.info(
                f"Committed transaction: {self.current_transaction.transaction_id}"
            )

            # Clear current transaction
            completed_transaction = self.current_transaction
            self.current_transaction = None

            return completed_transaction.status == "committed"

        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            return False

    def rollback_transaction(self) -> bool:
        """Rollback the current transaction.

        Returns:
            True if rollback successful
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        try:
            # Rollback operations in reverse order
            successful_operations = [
                op for op in self.current_transaction.operations if op.success
            ]

            if successful_operations:
                rolled_back = self.manipulator.rollback_operations(
                    successful_operations
                )
                logger.info(f"Rolled back {rolled_back} operations")

            self.current_transaction.status = "rolled_back"
            self.current_transaction.end_time = datetime.now().isoformat()

            # Save final transaction state
            self._save_transaction_log(self.current_transaction)

            # Add to history
            self.transaction_history.append(self.current_transaction)

            logger.info(
                f"Rolled back transaction: {self.current_transaction.transaction_id}"
            )

            # Clear current transaction
            self.current_transaction = None

            return True

        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
            return False

    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a transaction.

        Args:
            transaction_id: Transaction ID

        Returns:
            Transaction status or None if not found
        """
        # Check current transaction
        if (
            self.current_transaction
            and self.current_transaction.transaction_id == transaction_id
        ):
            return self._transaction_to_dict(self.current_transaction)

        # Check history
        for txn in self.transaction_history:
            if txn.transaction_id == transaction_id:
                return self._transaction_to_dict(txn)

        # Check saved logs
        log_file = self.log_dir / f"{transaction_id}.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                return json.load(f)

        return None

    def list_transactions(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List transactions.

        Args:
            status: Filter by status
            limit: Maximum number of transactions to return

        Returns:
            List of transaction summaries
        """
        transactions = []

        # Add current transaction if exists
        if self.current_transaction:
            if not status or self.current_transaction.status == status:
                transactions.append(self._transaction_summary(self.current_transaction))

        # Add from history
        for txn in reversed(self.transaction_history):
            if not status or txn.status == status:
                transactions.append(self._transaction_summary(txn))

            if len(transactions) >= limit:
                break

        # Add from saved logs if needed
        if len(transactions) < limit:
            for log_file in sorted(self.log_dir.glob("txn_*.json"), reverse=True):
                try:
                    with open(log_file, "r") as f:
                        txn_data = json.load(f)

                    if not status or txn_data.get("status") == status:
                        transactions.append(
                            {
                                "transaction_id": txn_data["transaction_id"],
                                "status": txn_data["status"],
                                "start_time": txn_data["start_time"],
                                "end_time": txn_data.get("end_time"),
                                "operation_count": len(txn_data.get("operations", [])),
                            }
                        )

                    if len(transactions) >= limit:
                        break

                except Exception as e:
                    logger.warning(f"Failed to read transaction log {log_file}: {e}")

        return transactions[:limit]

    def recover_transaction(self, transaction_id: str) -> bool:
        """Recover and rollback an incomplete transaction.

        Args:
            transaction_id: Transaction ID to recover

        Returns:
            True if recovery successful
        """
        transaction_data = self.get_transaction_status(transaction_id)

        if not transaction_data:
            logger.error(f"Transaction not found: {transaction_id}")
            return False

        if transaction_data["status"] != "pending":
            logger.info(
                f"Transaction {transaction_id} is already {transaction_data['status']}"
            )
            return True

        # Reconstruct transaction object
        transaction = Transaction(
            transaction_id=transaction_data["transaction_id"],
            start_time=transaction_data["start_time"],
            status=transaction_data["status"],
            operations=[],
            metadata=transaction_data.get("metadata", {}),
        )

        # Reconstruct operations
        for op_data in transaction_data.get("operations", []):
            operation = FileOperation(
                operation_type=op_data["operation_type"],
                source_path=op_data["source_path"],
                target_path=op_data["target_path"],
                timestamp=op_data["timestamp"],
                success=op_data["success"],
                error=op_data.get("error"),
                rollback_info=op_data.get("rollback_info"),
            )
            transaction.operations.append(operation)

        # Rollback the transaction
        try:
            successful_ops = [op for op in transaction.operations if op.success]
            if successful_ops:
                rolled_back = self.manipulator.rollback_operations(successful_ops)
                logger.info(
                    f"Recovered transaction {transaction_id}: rolled back {rolled_back} operations"
                )

            transaction.status = "rolled_back"
            transaction.end_time = datetime.now().isoformat()
            self._save_transaction_log(transaction)

            return True

        except Exception as e:
            logger.error(f"Failed to recover transaction {transaction_id}: {e}")
            return False

    def cleanup_old_transactions(self, days: int = 30) -> int:
        """Clean up old transaction logs.

        Args:
            days: Delete logs older than this many days

        Returns:
            Number of logs deleted
        """
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted = 0

        for log_file in self.log_dir.glob("txn_*.json"):
            try:
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete old log {log_file}: {e}")

        logger.info(f"Deleted {deleted} old transaction logs")
        return deleted

    def _save_transaction_log(self, transaction: Transaction):
        """Save transaction to log file.

        Args:
            transaction: Transaction to save
        """
        log_file = self.log_dir / f"{transaction.transaction_id}.json"

        try:
            with open(log_file, "w") as f:
                json.dump(self._transaction_to_dict(transaction), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save transaction log: {e}")

    def _transaction_to_dict(self, transaction: Transaction) -> Dict[str, Any]:
        """Convert transaction to dictionary.

        Args:
            transaction: Transaction object

        Returns:
            Dictionary representation
        """
        return {
            "transaction_id": transaction.transaction_id,
            "start_time": transaction.start_time,
            "end_time": transaction.end_time,
            "status": transaction.status,
            "operations": [asdict(op) for op in transaction.operations],
            "metadata": transaction.metadata,
        }

    def _transaction_summary(self, transaction: Transaction) -> Dict[str, Any]:
        """Create transaction summary.

        Args:
            transaction: Transaction object

        Returns:
            Summary dictionary
        """
        successful_ops = sum(1 for op in transaction.operations if op.success)
        failed_ops = sum(1 for op in transaction.operations if not op.success)

        return {
            "transaction_id": transaction.transaction_id,
            "status": transaction.status,
            "start_time": transaction.start_time,
            "end_time": transaction.end_time,
            "operation_count": len(transaction.operations),
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
        }
