#!/usr/bin/env python3
"""
Demo script showing batch processing with queue management.
"""

import time
import tempfile
import json
from pathlib import Path
import threading

from src.utils.batch_processor import BatchProcessor, AsyncBatchProcessor
from src.utils.priority_queue import PersistentPriorityQueue, CircularBuffer
from src.utils.progress import ProgressTracker


# Create mock components for demo
class MockClaudeClient:
    def analyze_document(self, content):
        time.sleep(0.1)  # Simulate API call
        return {"analysis": "complete"}


class MockFileProcessor:
    def process_file(self, file_path):
        time.sleep(0.05)  # Simulate processing
        return type(
            "obj",
            (object,),
            {"success": True, "content": f"Content of {file_path}", "format": "txt"},
        )()


class MockMetadataExtractor:
    def extract_metadata(self, **kwargs):
        return type(
            "obj",
            (object,),
            {"to_dict": lambda: {"metadata": "extracted"}, "confidence_score": 0.9},
        )()


class MockOrganizationEngine:
    def determine_target_location(self, metadata):
        return "organized/documents", "default_rule"


def demo_basic_batch_processing():
    """Demonstrate basic batch processing."""
    print("=== Basic Batch Processing Demo ===\n")

    # Create batch processor with mock components
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        processor = BatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=MockFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=3,
            rate_limit=120,  # 2 per second
        )

        # Add files to queue
        print("Adding files to queue...")
        files = [f"/demo/file_{i}.txt" for i in range(10)]
        added = processor.add_to_queue(files, priority=1)
        print(f"Added {added} files to queue")

        # Check queue status
        status = processor.get_queue_status()
        print(f"\nQueue status: {status}")

        # Process one batch
        print("\nProcessing first batch...")
        results = processor.process_batch()

        print(f"Processed {len(results)} files:")
        for file_path, success, error in results:
            status = "✓" if success else "✗"
            print(f"  {status} {Path(file_path).name}")

        # Check updated status
        status = processor.get_queue_status()
        print(f"\nUpdated queue status: {status}")

        # Process all remaining files
        print("\nProcessing all remaining files...")
        summary = processor.process_all()

        print(f"\nProcessing summary:")
        print(f"  Total processed: {summary['total_processed']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Duration: {summary['duration']:.2f}s")
        print(f"  Rate: {summary['files_per_second']:.2f} files/s")

        Path(tmp_db.name).unlink()


def demo_priority_queue():
    """Demonstrate priority queue functionality."""
    print("\n\n=== Priority Queue Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        processor = BatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=MockFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=2,
        )

        # Add files with different priorities
        print("Adding files with different priorities...")

        # High priority files
        high_priority = ["/important/urgent_1.pdf", "/important/urgent_2.pdf"]
        processor.add_to_queue(high_priority, priority=1)

        # Normal priority files
        normal_priority = ["/docs/normal_1.txt", "/docs/normal_2.txt"]
        processor.add_to_queue(normal_priority, priority=5)

        # Low priority files
        low_priority = ["/archives/old_1.txt", "/archives/old_2.txt"]
        processor.add_to_queue(low_priority, priority=10)

        print("\nProcessing files in priority order...")

        # Process batches and show order
        batch_num = 1
        while processor.get_queue_size() > 0:
            print(f"\nBatch {batch_num}:")
            results = processor.process_batch()

            for file_path, success, _ in results:
                print(f"  Processed: {file_path}")

            batch_num += 1

        Path(tmp_db.name).unlink()


def demo_async_processing():
    """Demonstrate asynchronous batch processing."""
    print("\n\n=== Async Processing Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        async_processor = AsyncBatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=MockFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=2,
        )

        # Add files
        print("Adding files for async processing...")
        files = [f"/async/file_{i}.txt" for i in range(20)]
        async_processor.add_to_queue(files)

        # Set up progress tracking
        processed_count = [0]

        def progress_callback(results):
            processed_count[0] += len(results)
            print(f"\rProcessed: {processed_count[0]} files", end="", flush=True)

        # Start async processing
        print("\nStarting async processing...")
        async_processor.start_processing(callback=progress_callback)

        # Monitor progress
        start_time = time.time()
        while async_processor.get_queue_size() > 0 and time.time() - start_time < 10:
            time.sleep(0.5)

        # Stop processing
        async_processor.stop_processing()
        print(f"\n\nAsync processing completed: {processed_count[0]} files")

        Path(tmp_db.name).unlink()


def demo_error_handling_and_retry():
    """Demonstrate error handling and retry logic."""
    print("\n\n=== Error Handling and Retry Demo ===\n")

    # Create processor with error simulation
    class ErrorProneFileProcessor:
        def __init__(self):
            self.call_count = 0

        def process_file(self, file_path):
            self.call_count += 1
            # Fail first attempt, succeed on retry
            if "problematic" in file_path and self.call_count % 2 == 1:
                raise Exception("Simulated processing error")

            return type(
                "obj",
                (object,),
                {
                    "success": True,
                    "content": f"Content of {file_path}",
                    "format": "txt",
                },
            )()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        processor = BatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=ErrorProneFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=3,
            max_retries=2,
        )

        # Add mix of normal and problematic files
        print("Adding files (some will fail initially)...")
        files = [
            "/normal/file_1.txt",
            "/problematic/file_2.txt",
            "/normal/file_3.txt",
            "/problematic/file_4.txt",
        ]
        processor.add_to_queue(files)

        # First processing attempt
        print("\nFirst processing attempt...")
        results = processor.process_batch()

        for file_path, success, error in results:
            status = "✓" if success else "✗"
            error_msg = f" - {error}" if error else ""
            print(f"  {status} {Path(file_path).name}{error_msg}")

        # Check failed files
        failed_files = processor.get_failed_files()
        print(f"\nFailed files: {len(failed_files)}")
        for item in failed_files:
            print(f"  {item.file_path} - {item.error}")

        # Retry failed files
        print("\nRetrying failed files...")
        processor.retry_failed_files()

        # Process again
        results = processor.process_batch()

        print("\nRetry results:")
        for file_path, success, error in results:
            status = "✓" if success else "✗"
            print(f"  {status} {Path(file_path).name}")

        Path(tmp_db.name).unlink()


def demo_pause_resume():
    """Demonstrate pause and resume functionality."""
    print("\n\n=== Pause/Resume Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        processor = BatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=MockFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=1,
        )

        # Add files
        files = [f"/pausable/file_{i}.txt" for i in range(6)]
        processor.add_to_queue(files)

        print("Processing with pause/resume...")

        # Process some files
        for i in range(2):
            results = processor.process_batch()
            if results:
                print(f"Processed: {Path(results[0][0]).name}")

        # Pause processing
        print("\nPausing processing...")
        processor.pause()

        # Try to process while paused (should return empty)
        results = processor.process_batch()
        print(f"Processing while paused: {len(results)} files")

        # Resume processing
        print("\nResuming processing...")
        processor.resume()

        # Continue processing
        while processor.get_queue_size() > 0:
            results = processor.process_batch()
            if results:
                print(f"Processed: {Path(results[0][0]).name}")

        print("\nAll files processed")

        Path(tmp_db.name).unlink()


def demo_queue_statistics():
    """Demonstrate queue statistics and reporting."""
    print("\n\n=== Queue Statistics Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        processor = BatchProcessor(
            claude_client=MockClaudeClient(),
            file_processor=MockFileProcessor(),
            metadata_extractor=MockMetadataExtractor(),
            organization_engine=MockOrganizationEngine(),
            db_path=tmp_db.name,
            batch_size=3,
        )

        # Add various files
        print("Building queue with various priorities...")

        # Add files in batches with different priorities
        for priority in [1, 5, 10]:
            files = [f"/priority_{priority}/file_{i}.txt" for i in range(3)]
            processor.add_to_queue(files, priority=priority)

        # Initial statistics
        print("\nInitial queue statistics:")
        status = processor.get_queue_status()
        for state, count in status.items():
            print(f"  {state}: {count}")

        # Process some files
        print("\nProcessing partial batch...")
        processor.process_batch()

        # Updated statistics
        print("\nUpdated statistics:")
        status = processor.get_queue_status()
        for state, count in status.items():
            print(f"  {state}: {count}")

        # Export queue state
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_json:
            processor.export_queue_state(tmp_json.name)

            # Show exported state
            with open(tmp_json.name, "r") as f:
                state = json.load(f)

            print(f"\nExported queue state:")
            print(f"  Total items: {state['queue_size']}")
            print(f"  Export time: {state['export_time']}")
            print(f"  Status summary: {state['status_summary']}")

            Path(tmp_json.name).unlink()

        # Processing history
        print("\nProcessing history:")
        history = processor.get_processing_history(limit=5)
        for record in history:
            print(f"  {record['file_path']} - {record['status']}")

        Path(tmp_db.name).unlink()


def main():
    """Run all demos."""
    print("Batch Processing Demo")
    print("====================")

    demo_basic_batch_processing()
    demo_priority_queue()
    demo_async_processing()
    demo_error_handling_and_retry()
    demo_pause_resume()
    demo_queue_statistics()

    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
