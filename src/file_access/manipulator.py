"""
File manipulation service for organizing files.
"""

import os
import shutil
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class FileOperation:
    """Represents a file operation."""

    operation_type: str  # 'copy' or 'move'
    source_path: str
    target_path: str
    timestamp: str
    success: bool
    error: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None


class FileManipulator:
    """Service for moving and copying files to organized locations."""

    def __init__(
        self,
        base_directory: str,
        dry_run: bool = False,
        verify_integrity: bool = True,
        conflict_strategy: str = "rename",
    ):
        """Initialize file manipulator.

        Args:
            base_directory: Base directory for organized files
            dry_run: Whether to simulate operations without actual changes
            verify_integrity: Whether to verify file integrity after operations
            conflict_strategy: How to handle naming conflicts
                - "rename": Add suffix to conflicting names
                - "skip": Skip files with conflicts
                - "overwrite": Overwrite existing files
                - "ask": Ask user for each conflict
        """
        self.base_directory = Path(base_directory)
        self.dry_run = dry_run
        self.verify_integrity = verify_integrity
        self.conflict_strategy = conflict_strategy
        self.operations_log = []
        self.pending_operations = []
        self.completed_operations = []

        # Create base directory if it doesn't exist
        if not self.dry_run:
            self.base_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FileManipulator initialized: base_dir={base_directory}, dry_run={dry_run}"
        )

    def organize_file(
        self, source_path: str, target_relative_path: str, operation: str = "move"
    ) -> bool:
        """Organize a single file.

        Args:
            source_path: Path to source file
            target_relative_path: Relative path within base directory
            operation: 'move' or 'copy'

        Returns:
            True if operation successful
        """
        source = Path(source_path)

        if not source.exists():
            logger.error(f"Source file not found: {source_path}")
            self._log_operation(
                FileOperation(
                    operation_type=operation,
                    source_path=source_path,
                    target_path=target_relative_path,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error="Source file not found",
                )
            )
            return False

        # Construct full target path
        target = self.base_directory / target_relative_path

        # Handle naming conflicts
        final_target = self._handle_conflict(target)

        # If conflict resolution returned None (skip), return early
        if final_target is None:
            return False

        # Create directory structure
        if not self._create_directory(final_target.parent):
            return False

        # Perform operation
        if operation == "move":
            return self._move_file(source, final_target)
        elif operation == "copy":
            return self._copy_file(source, final_target)
        else:
            logger.error(f"Unknown operation: {operation}")
            return False

    def _move_file(self, source: Path, target: Path) -> bool:
        """Move a file.

        Args:
            source: Source file path
            target: Target file path

        Returns:
            True if successful
        """
        operation = FileOperation(
            operation_type="move",
            source_path=str(source),
            target_path=str(target),
            timestamp=datetime.now().isoformat(),
            success=False,
        )

        if self.dry_run:
            operation.success = True
            self._log_operation(operation)
            logger.info(f"[DRY RUN] Would move: {source} -> {target}")
            return True

        try:
            # Calculate checksum before move (if integrity check enabled)
            checksum_before = None
            if self.verify_integrity:
                checksum_before = self._calculate_checksum(source)

            # Perform move
            shutil.move(str(source), str(target))

            # Verify integrity
            if self.verify_integrity and checksum_before:
                checksum_after = self._calculate_checksum(target)
                if checksum_before != checksum_after:
                    raise ValueError("File integrity check failed after move")

            operation.success = True
            operation.rollback_info = {
                "original_path": str(source),
                "moved_to": str(target),
            }

            logger.info(f"Moved: {source} -> {target}")

        except Exception as e:
            operation.error = str(e)
            logger.error(f"Failed to move {source}: {e}")

        self._log_operation(operation)
        return operation.success

    def _copy_file(self, source: Path, target: Path) -> bool:
        """Copy a file.

        Args:
            source: Source file path
            target: Target file path

        Returns:
            True if successful
        """
        operation = FileOperation(
            operation_type="copy",
            source_path=str(source),
            target_path=str(target),
            timestamp=datetime.now().isoformat(),
            success=False,
        )

        if self.dry_run:
            operation.success = True
            self._log_operation(operation)
            logger.info(f"[DRY RUN] Would copy: {source} -> {target}")
            return True

        try:
            # Calculate checksum before copy (if integrity check enabled)
            checksum_before = None
            if self.verify_integrity:
                checksum_before = self._calculate_checksum(source)

            # Perform copy
            shutil.copy2(str(source), str(target))

            # Verify integrity
            if self.verify_integrity and checksum_before:
                checksum_after = self._calculate_checksum(target)
                if checksum_before != checksum_after:
                    raise ValueError("File integrity check failed after copy")

            operation.success = True
            operation.rollback_info = {"created_file": str(target)}

            logger.info(f"Copied: {source} -> {target}")

        except Exception as e:
            operation.error = str(e)
            logger.error(f"Failed to copy {source}: {e}")

        self._log_operation(operation)
        return operation.success

    def _create_directory(self, directory: Path) -> bool:
        """Create directory if it doesn't exist.

        Args:
            directory: Directory path

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create directory: {directory}")
            return True

        try:
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False

    def _handle_conflict(self, target: Path) -> Path:
        """Handle naming conflicts based on strategy.

        Args:
            target: Target file path

        Returns:
            Final target path after conflict resolution
        """
        if not target.exists():
            return target

        if self.conflict_strategy == "overwrite":
            logger.warning(f"Overwriting existing file: {target}")
            return target

        elif self.conflict_strategy == "skip":
            logger.warning(f"Skipping file due to conflict: {target}")
            return None

        elif self.conflict_strategy == "rename":
            # Find unique name by adding suffix
            base = target.stem
            ext = target.suffix
            parent = target.parent
            counter = 1

            while True:
                new_name = f"{base}_{counter}{ext}"
                new_target = parent / new_name

                if not new_target.exists():
                    logger.info(f"Resolved conflict: {target} -> {new_target}")
                    return new_target

                counter += 1

                if counter > 1000:  # Safety limit
                    raise ValueError(f"Too many naming conflicts for {target}")

        elif self.conflict_strategy == "ask":
            return self._ask_user_conflict_resolution(target)

        else:
            raise ValueError(f"Unknown conflict strategy: {self.conflict_strategy}")

    def _ask_user_conflict_resolution(self, target: Path) -> Optional[Path]:
        """Ask user how to resolve naming conflict.

        Args:
            target: Conflicting target path

        Returns:
            Resolved path or None to skip
        """
        print(f"\nFile already exists: {target}")
        print("Options:")
        print("1. Rename (add suffix)")
        print("2. Overwrite")
        print("3. Skip")

        while True:
            choice = input("Select option (1-3): ").strip()

            if choice == "1":
                # Rename
                counter = 1
                while True:
                    new_name = f"{target.stem}_{counter}{target.suffix}"
                    new_target = target.parent / new_name

                    if not new_target.exists():
                        return new_target
                    counter += 1

            elif choice == "2":
                # Overwrite
                return target

            elif choice == "3":
                # Skip
                return None

            else:
                print("Invalid choice. Please select 1-3.")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Checksum string
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def batch_organize(
        self,
        files_with_paths: List[Tuple[str, str]],
        operation: str = "move",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Organize multiple files in batch.

        Args:
            files_with_paths: List of (source_path, target_relative_path) tuples
            operation: 'move' or 'copy'
            progress_callback: Optional callback for progress updates

        Returns:
            Summary of operations
        """
        total_files = len(files_with_paths)
        successful = 0
        failed = 0
        skipped = 0

        for i, (source, target) in enumerate(files_with_paths):
            result = self.organize_file(source, target, operation)

            if result:
                successful += 1
            else:
                # Check if it was skipped due to conflict
                if (
                    self.operations_log
                    and self.operations_log[-1].error == "Conflict - skipped"
                ):
                    skipped += 1
                else:
                    failed += 1

            if progress_callback:
                progress_callback(i + 1, total_files)

        return {
            "total": total_files,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "operations": self.operations_log[-total_files:],
        }

    def rollback_operations(
        self, operations: Optional[List[FileOperation]] = None
    ) -> int:
        """Rollback file operations.

        Args:
            operations: Specific operations to rollback (defaults to all)

        Returns:
            Number of operations rolled back
        """
        if operations is None:
            operations = [
                op for op in self.operations_log if op.success and op.rollback_info
            ]

        rolled_back = 0

        for operation in reversed(operations):  # Reverse order for proper rollback
            if not operation.rollback_info:
                continue

            try:
                if operation.operation_type == "move":
                    # Reverse move operation
                    original = operation.rollback_info["original_path"]
                    current = operation.rollback_info["moved_to"]

                    if Path(current).exists() and not Path(original).exists():
                        shutil.move(current, original)
                        logger.info(f"Rolled back move: {current} -> {original}")
                        rolled_back += 1

                elif operation.operation_type == "copy":
                    # Remove copied file
                    created = operation.rollback_info["created_file"]

                    if Path(created).exists():
                        os.remove(created)
                        logger.info(f"Rolled back copy: removed {created}")
                        rolled_back += 1

            except Exception as e:
                logger.error(f"Failed to rollback operation: {e}")

        return rolled_back

    def _log_operation(self, operation: FileOperation):
        """Log an operation.

        Args:
            operation: Operation to log
        """
        self.operations_log.append(operation)

        if operation.success:
            self.completed_operations.append(operation)

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations.

        Returns:
            Summary dictionary
        """
        successful = sum(1 for op in self.operations_log if op.success)
        failed = sum(1 for op in self.operations_log if not op.success)

        return {
            "total_operations": len(self.operations_log),
            "successful": successful,
            "failed": failed,
            "operations_by_type": {
                "move": sum(
                    1 for op in self.operations_log if op.operation_type == "move"
                ),
                "copy": sum(
                    1 for op in self.operations_log if op.operation_type == "copy"
                ),
            },
            "dry_run": self.dry_run,
        }

    def export_operations_log(self, output_path: str):
        """Export operations log to file.

        Args:
            output_path: Path for output file
        """
        log_data = [asdict(op) for op in self.operations_log]

        with open(output_path, "w") as f:
            json.dump(log_data, f, indent=2)

    def verify_organization(
        self, expected_structure: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Verify that files are organized as expected.

        Args:
            expected_structure: Dictionary mapping directories to expected files

        Returns:
            Verification results
        """
        results = {
            "verified": True,
            "missing_files": [],
            "unexpected_files": [],
            "missing_directories": [],
        }

        for directory, expected_files in expected_structure.items():
            dir_path = self.base_directory / directory

            if not dir_path.exists():
                results["missing_directories"].append(directory)
                results["verified"] = False
                continue

            # Check expected files
            for file_name in expected_files:
                file_path = dir_path / file_name
                if not file_path.exists():
                    results["missing_files"].append(str(file_path))
                    results["verified"] = False

            # Check for unexpected files
            actual_files = {f.name for f in dir_path.iterdir() if f.is_file()}
            expected_set = set(expected_files)
            unexpected = actual_files - expected_set

            for unexpected_file in unexpected:
                results["unexpected_files"].append(str(dir_path / unexpected_file))

        return results

    def create_organization_report(self) -> str:
        """Create a detailed report of organization operations.

        Returns:
            Report as string
        """
        report = []
        report.append("File Organization Report")
        report.append("=" * 30)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Base Directory: {self.base_directory}")
        report.append(f"Dry Run: {self.dry_run}")
        report.append("")

        # Summary
        summary = self.get_operation_summary()
        report.append("Summary:")
        report.append(f"  Total Operations: {summary['total_operations']}")
        report.append(f"  Successful: {summary['successful']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  Moves: {summary['operations_by_type']['move']}")
        report.append(f"  Copies: {summary['operations_by_type']['copy']}")
        report.append("")

        # Successful operations
        report.append("Successful Operations:")
        for op in self.completed_operations:
            report.append(
                f"  {op.operation_type}: {Path(op.source_path).name} -> {op.target_path}"
            )

        # Failed operations
        failed_ops = [op for op in self.operations_log if not op.success]
        if failed_ops:
            report.append("")
            report.append("Failed Operations:")
            for op in failed_ops:
                report.append(f"  {op.operation_type}: {Path(op.source_path).name}")
                report.append(f"    Error: {op.error}")

        return "\n".join(report)
