"""
Demo script showing error handling and file transaction features.
"""

import logging
import sys
from pathlib import Path
from time import sleep

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.error_handler import ErrorHandler, ErrorType
from src.utils.file_transaction import FileTransaction, OperationType
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demo ===\n")

    error_handler = ErrorHandler(max_retries=3, initial_backoff=1.0)

    # 1. Handle transient error with retry
    print("1. Simulating transient error with retry:")
    call_count = 0

    def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Connection failed (attempt {call_count})")
        return f"Success on attempt {call_count}"

    result, error = error_handler.handle_error(
        ConnectionError("Initial connection error"),
        "flaky_operation",
        retry_func=flaky_operation,
    )
    print(f"Result: {result}")
    print(f"Final error: {error}\n")

    # 2. Handle file access error
    print("2. Simulating file access error:")
    try:
        with open("/nonexistent/file.txt") as f:
            f.read()
    except FileNotFoundError as e:
        result, error = error_handler.handle_error(e, "file_read_operation")
        print(f"Handled file error: {error}\n")

    # 3. Handle API error
    print("3. Simulating API error:")
    from unittest.mock import Mock

    mock_request = Mock()
    api_error = anthropic.APIError(
        "API authentication failed", request=mock_request, body=None
    )
    result, error = error_handler.handle_error(api_error, "api_call_operation")
    print(f"Handled API error: {error}\n")

    # 4. Handle processing error
    print("4. Simulating processing error:")
    try:
        data = {"key": "value"}
        value = data["missing_key"]
    except KeyError as e:
        result, error = error_handler.handle_error(e, "data_processing_operation")
        print(f"Handled processing error: {error}\n")

    # 5. Display error statistics
    print("5. Error statistics:")
    stats = error_handler.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print("Error counts by type:")
    for error_type, count in stats["error_counts_by_type"].items():
        if count > 0:
            print(f"  {error_type}: {count}")
    print()

    # 6. Save error report
    print("6. Saving error report:")
    report_path = Path("error_report_demo.json")
    error_handler.save_error_report(report_path)
    print(f"Error report saved to {report_path}")


def demonstrate_file_transactions():
    """Demonstrate file transaction capabilities."""
    print("\n=== File Transaction Demo ===\n")

    # Create test directory
    test_dir = Path("transaction_demo")
    test_dir.mkdir(exist_ok=True)

    # 1. Successful transaction
    print("1. Demonstrating successful transaction:")
    transaction = FileTransaction()

    # Add operations
    new_dir = test_dir / "organized"
    transaction.add_mkdir(new_dir)

    # Create test files
    source_file = test_dir / "document.txt"
    source_file.write_text("Important document content")

    # Add operations to transaction
    transaction.add_copy(source_file, new_dir / "document_copy.txt")
    transaction.add_move(source_file, new_dir / "document_moved.txt")
    transaction.add_write(new_dir / "metadata.txt", "File metadata information")

    try:
        transaction.execute()
        print("Transaction executed successfully!")
        print(f"Created directory: {new_dir}")
        print(f"Files in directory: {list(new_dir.iterdir())}")
    except Exception as e:
        print(f"Transaction failed: {e}")

    # Save transaction log
    log_path = test_dir / "transaction_success.log"
    transaction.save_transaction_log(log_path)
    print(f"Transaction log saved to: {log_path}\n")

    # 2. Failed transaction with rollback
    print("2. Demonstrating failed transaction with rollback:")
    transaction2 = FileTransaction()

    # Create another test file
    source_file2 = test_dir / "document2.txt"
    source_file2.write_text("Another document")

    # Add operations including one that will fail
    transaction2.add_move(source_file2, new_dir / "document2_moved.txt")
    transaction2.add_delete(test_dir / "nonexistent_file.txt")  # This will fail

    try:
        transaction2.execute()
        print("Transaction executed successfully!")
    except Exception as e:
        print(f"Transaction failed: {e}")
        print("Rolling back changes...")
        # Check that source file still exists (rollback worked)
        if source_file2.exists():
            print(f"Rollback successful: {source_file2} still exists")
        else:
            print("Rollback failed!")

    # Save transaction log
    log_path2 = test_dir / "transaction_failed.log"
    transaction2.save_transaction_log(log_path2)
    print(f"Transaction log saved to: {log_path2}\n")

    # 3. Complex transaction
    print("3. Demonstrating complex transaction with multiple operations:")
    transaction3 = FileTransaction()

    # Create multiple test files
    for i in range(3):
        file_path = test_dir / f"file_{i}.txt"
        file_path.write_text(f"Content of file {i}")

    # Add multiple operations
    archive_dir = test_dir / "archive"
    transaction3.add_mkdir(archive_dir)

    for i in range(3):
        source = test_dir / f"file_{i}.txt"
        dest = archive_dir / f"archived_{i}.txt"
        transaction3.add_move(source, dest)

    # Create a summary file
    summary = "Files archived:\n"
    for i in range(3):
        summary += f"- archived_{i}.txt\n"
    transaction3.add_write(archive_dir / "summary.txt", summary)

    try:
        transaction3.execute()
        print("Complex transaction executed successfully!")
        print(f"Files in archive: {list(archive_dir.iterdir())}")
        print(f"Summary content: {(archive_dir / 'summary.txt').read_text()}")
    except Exception as e:
        print(f"Transaction failed: {e}")

    # Save transaction log
    log_path3 = test_dir / "transaction_complex.log"
    transaction3.save_transaction_log(log_path3)
    print(f"Transaction log saved to: {log_path3}")


def cleanup_demo_files():
    """Clean up demo files."""
    print("\n=== Cleaning up demo files ===")

    # Remove demo directories
    import shutil

    for path in ["transaction_demo", "error_report_demo.json"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
            print(f"Removed: {path}")


if __name__ == "__main__":
    try:
        demonstrate_error_handling()
        demonstrate_file_transactions()
    finally:
        # Cleanup
        cleanup_demo_files()

    print("\n=== Demo completed successfully! ===")
