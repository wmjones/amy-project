#!/usr/bin/env python3
"""
Demo script showing file manipulation service.
"""

import tempfile
import shutil
from pathlib import Path
import logging

from src.file_access.manipulator import FileManipulator
from src.file_access.transaction import TransactionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_basic_file_operations():
    """Demonstrate basic file operations."""
    print("=== Basic File Operations Demo ===\n")

    # Create test environment
    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as target_dir:
            # Create test files
            test_files = []
            for i in range(5):
                file_path = Path(source_dir) / f"document_{i}.pdf"
                file_path.write_text(f"Document {i} content")
                test_files.append(file_path)

            # Create manipulator
            manipulator = FileManipulator(
                base_directory=target_dir,
                dry_run=False,
                verify_integrity=True,
                conflict_strategy="rename",
            )

            # Organize files
            print("Organizing files...")

            # Move some files
            manipulator.organize_file(
                str(test_files[0]), "Documents/2024/Q1/report.pdf", operation="move"
            )

            manipulator.organize_file(
                str(test_files[1]), "Documents/2024/Q1/analysis.pdf", operation="move"
            )

            # Copy some files
            manipulator.organize_file(
                str(test_files[2]), "Backup/document_2.pdf", operation="copy"
            )

            # Show results
            summary = manipulator.get_operation_summary()
            print(f"\nOperation Summary:")
            print(f"  Total: {summary['total_operations']}")
            print(f"  Successful: {summary['successful']}")
            print(f"  Failed: {summary['failed']}")

            # Generate report
            report = manipulator.create_organization_report()
            print(f"\n{report}")


def demo_dry_run_mode():
    """Demonstrate dry run mode."""
    print("\n\n=== Dry Run Mode Demo ===\n")

    with tempfile.TemporaryDirectory() as source_dir:
        # Create test files
        file1 = Path(source_dir) / "important.docx"
        file2 = Path(source_dir) / "backup.zip"
        file1.write_text("Important document")
        file2.write_text("Backup data")

        # Create manipulator in dry run mode
        manipulator = FileManipulator(base_directory="/tmp/organized", dry_run=True)

        print("Simulating file organization (dry run)...")

        # Simulate operations
        manipulator.organize_file(
            str(file1), "Documents/Important/important.docx", operation="move"
        )

        manipulator.organize_file(
            str(file2), "Backups/2024/backup.zip", operation="copy"
        )

        # Verify files weren't actually moved
        print(f"\nOriginal files still exist:")
        print(f"  {file1.name}: {file1.exists()}")
        print(f"  {file2.name}: {file2.exists()}")

        # Show what would have happened
        for op in manipulator.operations_log:
            print(f"\n[DRY RUN] Would {op.operation_type}:")
            print(f"  From: {op.source_path}")
            print(f"  To: {op.target_path}")


def demo_conflict_resolution():
    """Demonstrate conflict resolution strategies."""
    print("\n\n=== Conflict Resolution Demo ===\n")

    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as target_dir:
            # Create test files with same name
            files = []
            for i in range(3):
                file_path = Path(source_dir) / f"file_{i}" / "report.pdf"
                file_path.parent.mkdir(exist_ok=True)
                file_path.write_text(f"Report version {i}")
                files.append(file_path)

            # Test rename strategy
            print("Testing 'rename' strategy:")
            manipulator = FileManipulator(
                base_directory=target_dir, conflict_strategy="rename"
            )

            for file in files:
                manipulator.organize_file(
                    str(file), "Reports/monthly_report.pdf", operation="copy"
                )

            # Check results
            reports_dir = Path(target_dir) / "Reports"
            created_files = list(reports_dir.glob("*.pdf"))
            print(f"Created files: {[f.name for f in created_files]}")

            # Test skip strategy
            print("\n\nTesting 'skip' strategy:")
            skip_dir = Path(target_dir) / "skip_test"
            manipulator_skip = FileManipulator(
                base_directory=skip_dir, conflict_strategy="skip"
            )

            # First file should succeed
            result1 = manipulator_skip.organize_file(
                str(files[0]), "report.pdf", operation="copy"
            )

            # Second file should be skipped
            result2 = manipulator_skip.organize_file(
                str(files[1]), "report.pdf", operation="copy"
            )

            print(f"First file: {'Success' if result1 else 'Failed'}")
            print(f"Second file: {'Success' if result2 else 'Skipped'}")


def demo_batch_operations():
    """Demonstrate batch file operations."""
    print("\n\n=== Batch Operations Demo ===\n")

    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as target_dir:
            # Create many test files
            files_with_paths = []

            # Create invoices
            for i in range(5):
                file_path = Path(source_dir) / f"invoice_{i}.pdf"
                file_path.write_text(f"Invoice {i}")
                files_with_paths.append(
                    (str(file_path), f"Financial/Invoices/2024/invoice_{i}.pdf")
                )

            # Create reports
            for i in range(3):
                file_path = Path(source_dir) / f"report_{i}.docx"
                file_path.write_text(f"Report {i}")
                files_with_paths.append(
                    (str(file_path), f"Documents/Reports/Q1/report_{i}.docx")
                )

            # Create manipulator
            manipulator = FileManipulator(base_directory=target_dir)

            # Define progress callback
            def progress_callback(current, total):
                print(f"Progress: {current}/{total} ({current/total*100:.0f}%)")

            # Batch organize
            print("Organizing files in batch...")
            results = manipulator.batch_organize(
                files_with_paths, operation="move", progress_callback=progress_callback
            )

            print(f"\nBatch Results:")
            print(f"  Total: {results['total']}")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            print(f"  Skipped: {results['skipped']}")


def demo_transaction_management():
    """Demonstrate transaction-based operations."""
    print("\n\n=== Transaction Management Demo ===\n")

    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as target_dir:
            # Create test files
            files = []
            for i in range(4):
                file_path = Path(source_dir) / f"data_{i}.csv"
                file_path.write_text(f"Data {i}")
                files.append(file_path)

            # Create managers
            manipulator = FileManipulator(base_directory=target_dir)
            transaction_manager = TransactionManager(manipulator)

            # Begin transaction
            print("Starting transaction...")
            txn_id = transaction_manager.begin_transaction(
                {"operation": "Monthly data organization", "user": "demo_user"}
            )

            print(f"Transaction ID: {txn_id}")

            # Add operations to transaction
            print("\nAdding operations to transaction...")

            transaction_manager.add_operation(
                str(files[0]), "Data/2024/January/sales.csv", operation_type="move"
            )

            transaction_manager.add_operation(
                str(files[1]), "Data/2024/January/inventory.csv", operation_type="move"
            )

            # Simulate error with third file
            transaction_manager.add_operation(
                "/nonexistent/file.csv",
                "Data/2024/January/missing.csv",
                operation_type="move",
            )

            # Check transaction status
            status = transaction_manager.get_transaction_status(txn_id)
            print(f"\nTransaction Status: {status['status']}")
            print(f"Operations: {len(status['operations'])}")

            # Decide whether to commit or rollback
            failed_ops = [op for op in status["operations"] if not op["success"]]

            if failed_ops:
                print(f"\nFound {len(failed_ops)} failed operations. Rolling back...")
                transaction_manager.rollback_transaction()

                # Verify files are back
                for file in files[:2]:
                    print(
                        f"  {file.name}: {'Restored' if file.exists() else 'Missing'}"
                    )
            else:
                print("\nAll operations successful. Committing...")
                transaction_manager.commit_transaction()

            # Show transaction history
            print("\nTransaction History:")
            for txn in transaction_manager.list_transactions():
                print(f"  {txn['transaction_id']}: {txn['status']}")


def demo_verification_and_reporting():
    """Demonstrate verification and reporting."""
    print("\n\n=== Verification and Reporting Demo ===\n")

    with tempfile.TemporaryDirectory() as target_dir:
        # Create organized structure
        manipulator = FileManipulator(base_directory=target_dir)

        # Create some files directly (simulating previous organization)
        structure = {
            "Documents/2024": ["report.pdf", "analysis.docx"],
            "Financial/Invoices": ["inv_001.pdf", "inv_002.pdf"],
            "Projects/Alpha": ["spec.doc", "timeline.xlsx"],
        }

        for dir_path, files in structure.items():
            dir_full = Path(target_dir) / dir_path
            dir_full.mkdir(parents=True, exist_ok=True)

            for file_name in files:
                file_path = dir_full / file_name
                file_path.write_text(f"Content of {file_name}")

        # Define expected structure
        expected_structure = {
            "Documents/2024": [
                "report.pdf",
                "analysis.docx",
                "summary.pdf",
            ],  # summary.pdf is missing
            "Financial/Invoices": ["inv_001.pdf", "inv_002.pdf"],
            "Projects/Alpha": ["spec.doc", "timeline.xlsx"],
            "Projects/Beta": ["plan.doc"],  # This directory doesn't exist
        }

        # Verify organization
        print("Verifying file organization...")
        verification = manipulator.verify_organization(expected_structure)

        print(f"\nVerification Results:")
        print(f"  Verified: {verification['verified']}")
        print(f"  Missing Files: {len(verification['missing_files'])}")
        for missing in verification["missing_files"]:
            print(f"    - {missing}")
        print(f"  Missing Directories: {len(verification['missing_directories'])}")
        for missing_dir in verification["missing_directories"]:
            print(f"    - {missing_dir}")

        # Create operation report
        print("\n\nOrganization Report:")
        print("=" * 40)

        # Simulate some operations for the report
        test_file = Path(target_dir) / "test.txt"
        test_file.write_text("Test")

        manipulator.organize_file(str(test_file), "Archive/test.txt")
        manipulator.organize_file("/nonexistent.txt", "Archive/missing.txt")

        report = manipulator.create_organization_report()
        print(report)


def main():
    """Run all demos."""
    print("File Manipulation Service Demo")
    print("==============================")

    demo_basic_file_operations()
    demo_dry_run_mode()
    demo_conflict_resolution()
    demo_batch_operations()
    demo_transaction_management()
    demo_verification_and_reporting()

    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
