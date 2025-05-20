"""
Unit tests for progress tracking and statistics.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.progress import ProgressTracker, ConsoleProgressBar, FileStatus
from src.utils.statistics import StatisticsCollector, FileStatistics


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a progress tracker."""
        return ProgressTracker(
            total_files=10, console_output=False  # Disable console output for tests
        )

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.total_files == 10
        assert tracker.processed_files == 0
        assert tracker.successful_files == 0
        assert tracker.failed_files == 0
        assert tracker.skipped_files == 0

    def test_update_progress_success(self, tracker):
        """Test updating progress with success."""
        tracker.start_file("/path/file1.txt")
        tracker.update_progress("/path/file1.txt", "success")

        assert tracker.processed_files == 1
        assert tracker.successful_files == 1
        assert tracker.failed_files == 0
        assert tracker.skipped_files == 0

    def test_update_progress_error(self, tracker):
        """Test updating progress with error."""
        tracker.start_file("/path/file2.txt")
        tracker.update_progress("/path/file2.txt", "error", error="File not found")

        assert tracker.processed_files == 1
        assert tracker.successful_files == 0
        assert tracker.failed_files == 1
        assert tracker.skipped_files == 0
        assert len(tracker.errors) == 1
        assert tracker.errors[0] == ("/path/file2.txt", "File not found")

    def test_update_progress_skipped(self, tracker):
        """Test updating progress with skip."""
        tracker.start_file("/path/file3.txt")
        tracker.update_progress("/path/file3.txt", "skipped")

        assert tracker.processed_files == 1
        assert tracker.successful_files == 0
        assert tracker.failed_files == 0
        assert tracker.skipped_files == 1

    def test_processing_time_tracking(self, tracker):
        """Test processing time tracking."""
        tracker.start_file("/path/file.txt")
        time.sleep(0.1)  # Simulate processing
        tracker.update_progress("/path/file.txt", "success")

        assert len(tracker.processing_times) == 1
        assert tracker.processing_times[0] >= 0.1

    def test_get_current_stats(self, tracker):
        """Test getting current statistics."""
        # Process some files
        for i in range(3):
            tracker.start_file(f"/path/file{i}.txt")
            status = "success" if i < 2 else "error"
            error = "Test error" if status == "error" else None
            tracker.update_progress(f"/path/file{i}.txt", status, error=error)

        stats = tracker.get_current_stats()

        assert stats["total_files"] == 10
        assert stats["processed_files"] == 3
        assert stats["successful_files"] == 2
        assert stats["failed_files"] == 1
        assert stats["percent_complete"] == 30.0
        assert stats["errors_count"] == 1

    def test_progress_callback(self):
        """Test progress callback functionality."""
        callback_data = []

        def progress_callback(stats):
            callback_data.append(stats)

        tracker = ProgressTracker(
            total_files=5, console_output=False, progress_callback=progress_callback
        )

        tracker.start_file("/path/file.txt")
        tracker.update_progress("/path/file.txt", "success")

        assert len(callback_data) == 1
        assert callback_data[0]["processed_files"] == 1

    def test_finish_and_summary(self, tracker):
        """Test finishing and getting summary."""
        # Process files
        tracker.start_file("/path/file1.txt")
        tracker.update_progress("/path/file1.txt", "success")

        tracker.finish()
        summary = tracker.get_summary()

        assert summary["total_files"] == 10
        assert summary["processed_files"] == 1
        assert summary["successful_files"] == 1
        assert "start_time" in summary
        assert "end_time" in summary
        assert "average_speed" in summary

    def test_save_report(self, tracker):
        """Test saving report to file."""
        # Process a file
        tracker.start_file("/path/file.txt")
        tracker.update_progress("/path/file.txt", "success", details={"size": 1024})
        tracker.finish()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            tracker.save_report(f.name)

        # Verify report
        with open(f.name, "r") as f:
            report = json.load(f)

        assert "summary" in report
        assert "file_statuses" in report
        assert len(report["file_statuses"]) == 1
        assert report["file_statuses"][0]["status"] == "success"

        Path(f.name).unlink()

    def test_error_report_generation(self, tracker):
        """Test error report generation."""
        # Add some errors
        tracker.update_progress("/path/file1.txt", "error", error="Permission denied")
        tracker.update_progress("/path/file2.txt", "error", error="Permission denied")
        tracker.update_progress("/path/file3.txt", "error", error="File not found")

        error_report = tracker.generate_error_report()

        assert "Error Report" in error_report
        assert "Permission denied" in error_report
        assert "File not found" in error_report

    @patch("sys.stdout")
    def test_console_display(self, mock_stdout):
        """Test console progress display."""
        tracker = ProgressTracker(total_files=100, console_output=True)

        tracker.start_file("/path/file.txt")
        tracker.update_progress("/path/file.txt", "success")

        # Check that something was printed
        mock_stdout.write.assert_called()


class TestStatisticsCollector:
    """Test StatisticsCollector functionality."""

    @pytest.fixture
    def collector(self):
        """Create a statistics collector."""
        return StatisticsCollector()

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.files_processed == 0
        assert collector.bytes_processed == 0
        assert len(collector.file_stats) == 0

    def test_add_file_stat(self, collector):
        """Test adding file statistics."""
        file_stat = FileStatistics(
            file_path="/path/file.txt",
            file_type="txt",
            file_size=1024,
            processing_time=0.5,
            operation="move",
            success=True,
            confidence_score=0.95,
        )

        collector.add_file_stat(file_stat)

        assert collector.files_processed == 1
        assert collector.bytes_processed == 1024
        assert len(collector.file_stats) == 1
        assert collector.operation_counts["move"] == 1
        assert len(collector.processing_times) == 1
        assert len(collector.confidence_scores) == 1

    def test_add_failed_file_stat(self, collector):
        """Test adding failed file statistics."""
        file_stat = FileStatistics(
            file_path="/path/file.txt",
            file_type="txt",
            file_size=1024,
            processing_time=0.5,
            operation="move",
            success=False,
            error="Permission denied",
        )

        collector.add_file_stat(file_stat)

        assert collector.files_processed == 1
        assert collector.bytes_processed == 0  # Failed files don't count
        assert collector.error_counts["str"] == 1  # Error type

    def test_calculate_statistics(self, collector):
        """Test statistics calculation."""
        # Add various file stats
        for i in range(5):
            file_stat = FileStatistics(
                file_path=f"/path/file{i}.txt",
                file_type="txt",
                file_size=1000 * (i + 1),
                processing_time=0.1 * (i + 1),
                operation="move" if i < 3 else "copy",
                success=i < 4,  # Last one fails
                confidence_score=0.8 + i * 0.05,
            )
            collector.add_file_stat(file_stat)

        collector.finish()
        stats = collector.calculate_statistics()

        # Verify summary
        assert stats["summary"]["total_files"] == 5
        assert stats["summary"]["successful_files"] == 4
        assert stats["summary"]["failed_files"] == 1

        # Verify performance metrics
        assert "performance" in stats
        assert stats["performance"]["avg_processing_time"] > 0
        assert stats["performance"]["files_per_second"] > 0

        # Verify operations
        assert stats["operations"]["move"] == 3
        assert stats["operations"]["copy"] == 2

    def test_file_type_statistics(self, collector):
        """Test file type statistics."""
        # Add different file types
        file_types = ["pdf", "pdf", "txt", "jpg", "jpg", "jpg"]
        for i, file_type in enumerate(file_types):
            file_stat = FileStatistics(
                file_path=f"/path/file{i}.{file_type}",
                file_type=file_type,
                file_size=1000,
                processing_time=0.1,
                operation="move",
                success=True,
            )
            collector.add_file_stat(file_stat)

        stats = collector.calculate_statistics()

        assert "file_types" in stats
        assert stats["file_types"]["jpg"]["count"] == 3
        assert stats["file_types"]["pdf"]["count"] == 2
        assert stats["file_types"]["txt"]["count"] == 1

    def test_generate_report(self, collector):
        """Test report generation."""
        # Add some stats
        file_stat = FileStatistics(
            file_path="/path/file.txt",
            file_type="txt",
            file_size=1024,
            processing_time=0.5,
            operation="move",
            success=True,
        )
        collector.add_file_stat(file_stat)
        collector.finish()

        report = collector.generate_report()

        assert "File Organization Statistics Report" in report
        assert "Summary" in report
        assert "Total Files: 1" in report

    def test_save_statistics(self, collector):
        """Test saving statistics to file."""
        # Add some stats
        file_stat = FileStatistics(
            file_path="/path/file.txt",
            file_type="txt",
            file_size=1024,
            processing_time=0.5,
            operation="move",
            success=True,
        )
        collector.add_file_stat(file_stat)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            collector.save_statistics(f.name)

        # Verify saved data
        with open(f.name, "r") as f:
            saved_stats = json.load(f)

        assert "summary" in saved_stats
        assert "files" in saved_stats
        assert len(saved_stats["files"]) == 1

        Path(f.name).unlink()

    def test_top_errors(self, collector):
        """Test getting top errors."""
        # Add various errors
        errors = [
            "Permission denied",
            "Permission denied",
            "File not found",
            "File not found",
            "File not found",
            "Disk full",
        ]

        for i, error in enumerate(errors):
            file_stat = FileStatistics(
                file_path=f"/path/file{i}.txt",
                file_type="txt",
                file_size=1024,
                processing_time=0.1,
                operation="move",
                success=False,
                error=error,
            )
            collector.add_file_stat(file_stat)

        top_errors = collector.get_top_errors(limit=2)

        assert len(top_errors) == 2
        assert top_errors[0]["error"] == "File not found"
        assert top_errors[0]["count"] == 3

    def test_performance_summary(self, collector):
        """Test performance summary."""
        # Add files with various processing times
        times = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

        for i, time_val in enumerate(times):
            file_stat = FileStatistics(
                file_path=f"/path/file{i}.txt",
                file_type="txt",
                file_size=1024,
                processing_time=time_val,
                operation="move",
                success=True,
            )
            collector.add_file_stat(file_stat)

        perf_summary = collector.get_performance_summary()

        assert "avg_processing_time" in perf_summary
        assert "median_processing_time" in perf_summary
        assert "p90_processing_time" in perf_summary
        assert "p99_processing_time" in perf_summary
        assert "slowest_files" in perf_summary
        assert "fastest_files" in perf_summary


class TestConsoleProgressBar:
    """Test ConsoleProgressBar functionality."""

    @patch("sys.stdout")
    def test_progress_bar_display(self, mock_stdout):
        """Test progress bar display."""
        progress = ConsoleProgressBar(total=100, description="Processing")

        progress.update(10)
        progress.update(20)

        # Check that output was written
        mock_stdout.write.assert_called()

    def test_progress_bar_completion(self):
        """Test progress bar completion."""
        progress = ConsoleProgressBar(total=50)

        for _ in range(50):
            progress.update(1)

        progress.finish()

        assert progress.current == progress.total
