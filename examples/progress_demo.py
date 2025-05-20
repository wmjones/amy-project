#!/usr/bin/env python3
"""
Demo script showing progress tracking and statistics collection.
"""

import time
import random
import tempfile
from pathlib import Path

from src.utils.progress import ProgressTracker, ConsoleProgressBar
from src.utils.statistics import StatisticsCollector, FileStatistics


def demo_basic_progress_tracking():
    """Demonstrate basic progress tracking."""
    print("=== Basic Progress Tracking Demo ===\n")

    # Create progress tracker
    total_files = 20
    tracker = ProgressTracker(total_files=total_files, console_output=True)

    # Simulate file processing
    file_types = ["pdf", "txt", "jpg", "docx"]
    statuses = ["success", "success", "success", "error", "skipped"]

    for i in range(total_files):
        file_path = f"/demo/file_{i}.{random.choice(file_types)}"

        # Start processing
        tracker.start_file(file_path)

        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))

        # Update status
        status = random.choice(statuses)
        error = None

        if status == "error":
            error = random.choice(
                ["Permission denied", "File not found", "Invalid format"]
            )

        tracker.update_progress(file_path, status, error=error)

    # Finish tracking
    tracker.finish()

    # Show summary
    print("\n\nSummary:")
    summary = tracker.get_summary()
    print(f"Total Files: {summary['processed_files']}")
    print(f"Successful: {summary['successful_files']}")
    print(f"Failed: {summary['failed_files']}")
    print(f"Skipped: {summary['skipped_files']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print(f"Average Speed: {summary['average_speed']:.2f} files/s")

    # Save report
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        tracker.save_report(f.name)
        print(f"\nReport saved to: {f.name}")


def demo_statistics_collection():
    """Demonstrate statistics collection."""
    print("\n\n=== Statistics Collection Demo ===\n")

    collector = StatisticsCollector()

    # Simulate processing various files
    file_data = [
        ("report.pdf", "pdf", 2048000, 1.2, "move", True, 0.92),
        ("image.jpg", "jpg", 512000, 0.3, "copy", True, 0.88),
        ("document.docx", "docx", 1024000, 0.8, "move", True, 0.95),
        ("video.mp4", "mp4", 10240000, 5.5, "copy", False, None),
        ("spreadsheet.xlsx", "xlsx", 768000, 0.6, "move", True, 0.90),
        ("presentation.pptx", "pptx", 3072000, 1.8, "move", True, 0.87),
        ("archive.zip", "zip", 5120000, 2.1, "copy", True, 0.85),
        ("text.txt", "txt", 10240, 0.05, "move", True, 0.99),
    ]

    print("Processing files...")
    for (
        file_name,
        file_type,
        size,
        time_val,
        operation,
        success,
        confidence,
    ) in file_data:
        file_stat = FileStatistics(
            file_path=f"/demo/{file_name}",
            file_type=file_type,
            file_size=size,
            processing_time=time_val,
            operation=operation,
            success=success,
            error="Processing failed" if not success else None,
            confidence_score=confidence,
        )

        collector.add_file_stat(file_stat)
        print(f"  {file_name}: {'✓' if success else '✗'}")

    # Finish collection
    collector.finish()

    # Generate report
    report = collector.generate_report()
    print(f"\n{report}")

    # Save detailed statistics
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        collector.save_statistics(f.name)
        print(f"\nDetailed statistics saved to: {f.name}")


def demo_progress_with_callback():
    """Demonstrate progress tracking with callback."""
    print("\n\n=== Progress with Callback Demo ===\n")

    progress_history = []

    def progress_callback(stats):
        """Callback to record progress."""
        progress_history.append(
            {
                "timestamp": time.time(),
                "percent": stats["percent_complete"],
                "files": stats["processed_files"],
            }
        )

    # Create tracker with callback
    tracker = ProgressTracker(
        total_files=10,
        console_output=False,  # Disable console for this demo
        progress_callback=progress_callback,
    )

    print("Processing files with callback...")
    for i in range(10):
        tracker.start_file(f"file_{i}.txt")
        time.sleep(0.2)
        tracker.update_progress(f"file_{i}.txt", "success")
        print(f"File {i+1}/10 - Progress: {progress_history[-1]['percent']:.1f}%")

    tracker.finish()

    print(f"\nCallback was triggered {len(progress_history)} times")
    print("Progress history:")
    for i, record in enumerate(progress_history[:5]):
        print(f"  {i+1}. {record['percent']:.1f}% ({record['files']} files)")
    if len(progress_history) > 5:
        print(f"  ... and {len(progress_history) - 5} more records")


def demo_console_progress_bar():
    """Demonstrate simple console progress bar."""
    print("\n\n=== Console Progress Bar Demo ===\n")

    # Demo 1: Simple counting
    print("Counting items:")
    progress = ConsoleProgressBar(total=50, description="Items")

    for i in range(50):
        time.sleep(0.05)
        progress.update(1)

    progress.finish()

    # Demo 2: Batch processing
    print("\n\nProcessing batches:")
    total_batches = 10
    items_per_batch = 100

    batch_progress = ConsoleProgressBar(total=total_batches, description="Batches")

    for batch in range(total_batches):
        # Process batch
        time.sleep(0.5)
        batch_progress.update(1)

    batch_progress.finish()


def demo_error_analysis():
    """Demonstrate error analysis and reporting."""
    print("\n\n=== Error Analysis Demo ===\n")

    tracker = ProgressTracker(total_files=30, console_output=False)

    # Simulate processing with various errors
    error_types = [
        "Permission denied",
        "File not found",
        "Invalid format",
        "Disk full",
        "Network error",
    ]

    print("Simulating file processing with errors...")
    for i in range(30):
        file_path = f"/demo/file_{i}.txt"
        tracker.start_file(file_path)

        # 30% chance of error
        if random.random() < 0.3:
            error = random.choice(error_types)
            tracker.update_progress(file_path, "error", error=error)
        else:
            tracker.update_progress(file_path, "success")

    tracker.finish()

    # Generate error report
    error_report = tracker.generate_error_report()
    print(f"\n{error_report}")

    # Show summary
    summary = tracker.get_summary()
    print(f"\nError Rate: {summary['failed_files']/summary['total_files']*100:.1f}%")


def demo_performance_analysis():
    """Demonstrate performance analysis."""
    print("\n\n=== Performance Analysis Demo ===\n")

    collector = StatisticsCollector()

    # Simulate files with varying processing times
    print("Processing files with different sizes...")

    for i in range(20):
        # Simulate larger files taking longer
        file_size = random.randint(1000, 10000000)
        processing_time = file_size / 1000000 * random.uniform(0.5, 1.5)

        file_stat = FileStatistics(
            file_path=f"/demo/file_{i}.dat",
            file_type="dat",
            file_size=file_size,
            processing_time=processing_time,
            operation="move",
            success=True,
            confidence_score=random.uniform(0.7, 1.0),
        )

        collector.add_file_stat(file_stat)

    collector.finish()

    # Get performance summary
    perf_summary = collector.get_performance_summary()

    print("\nPerformance Summary:")
    print(f"Average processing time: {perf_summary['avg_processing_time']:.3f}s")
    print(f"Median processing time: {perf_summary['median_processing_time']:.3f}s")
    print(f"90th percentile: {perf_summary['p90_processing_time']:.3f}s")
    print(f"99th percentile: {perf_summary['p99_processing_time']:.3f}s")

    print("\nSlowest files:")
    for i, file_info in enumerate(perf_summary["slowest_files"]):
        print(
            f"  {i+1}. {file_info['file']} - {file_info['time']:.3f}s ({file_info['size']} bytes)"
        )

    print("\nFastest files:")
    for i, file_info in enumerate(perf_summary["fastest_files"]):
        print(
            f"  {i+1}. {file_info['file']} - {file_info['time']:.3f}s ({file_info['size']} bytes)"
        )


def main():
    """Run all demos."""
    print("Progress Tracking and Statistics Demo")
    print("====================================")

    demo_basic_progress_tracking()
    demo_statistics_collection()
    demo_progress_with_callback()
    demo_console_progress_bar()
    demo_error_analysis()
    demo_performance_analysis()

    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
