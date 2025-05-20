"""
Progress tracking and logging system for file organization.
"""

import time
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from threading import Lock
import os


@dataclass
class FileStatus:
    """Status of a processed file."""

    file_path: str
    status: str  # 'success', 'error', 'skipped'
    timestamp: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Track progress of file organization operations."""

    def __init__(
        self,
        total_files: int,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        console_output: bool = True,
        progress_callback: Optional[Callable] = None,
    ):
        """Initialize progress tracker.

        Args:
            total_files: Total number of files to process
            log_file: Path to log file (optional)
            log_level: Logging level
            console_output: Whether to show console progress
            progress_callback: Optional callback for progress updates
        """
        self.total_files = total_files
        self.processed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        self.current_file = None
        self.start_time = time.time()
        self.end_time = None
        self.file_statuses = []
        self.errors = []
        self.console_output = console_output
        self.progress_callback = progress_callback
        self._lock = Lock()

        # Set up logging
        self._setup_logging(log_file, log_level)

        # Progress display settings
        self.last_update_time = 0
        self.update_interval = 0.1  # Update display every 0.1 seconds

        # Performance tracking
        self.processing_times = []
        self.current_file_start = None

        self.logger.info(f"Progress tracker initialized for {total_files} files")

    def _setup_logging(self, log_file: Optional[str], log_level: str):
        """Set up logging configuration."""
        self.logger = logging.getLogger("file_organizer")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            # Create log directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Set up rotating file handler
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

    def start_file(self, file_path: str):
        """Mark the start of processing a file.

        Args:
            file_path: Path to the file being processed
        """
        with self._lock:
            self.current_file = file_path
            self.current_file_start = time.time()
            self.logger.debug(f"Started processing: {file_path}")

    def update_progress(
        self,
        file_path: str,
        status: str,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Update progress for a file.

        Args:
            file_path: Path to the processed file
            status: Status ('success', 'error', 'skipped')
            error: Error message if status is 'error'
            details: Additional details about the operation
        """
        with self._lock:
            # Calculate processing time
            processing_time = None
            if self.current_file_start:
                processing_time = time.time() - self.current_file_start
                self.processing_times.append(processing_time)

            # Create file status
            file_status = FileStatus(
                file_path=file_path,
                status=status,
                timestamp=time.time(),
                error=error,
                details=details or {},
            )

            if processing_time:
                file_status.details["processing_time"] = processing_time

            self.file_statuses.append(file_status)

            # Update counters
            self.processed_files += 1

            if status == "success":
                self.successful_files += 1
                self.logger.info(f"Successfully processed: {file_path}")
            elif status == "error":
                self.failed_files += 1
                self.errors.append((file_path, error))
                self.logger.error(f"Failed to process: {file_path}, Error: {error}")
            elif status == "skipped":
                self.skipped_files += 1
                self.logger.warning(f"Skipped: {file_path}")

            # Reset current file
            self.current_file = None
            self.current_file_start = None

            # Update display
            self._update_display()

            # Call progress callback if provided
            if self.progress_callback:
                self.progress_callback(self.get_current_stats())

    def _update_display(self):
        """Update progress display."""
        current_time = time.time()

        # Throttle updates
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time

        if self.console_output:
            self._display_progress()

    def _display_progress(self):
        """Display progress in console."""
        with self._lock:
            percent = (
                (self.processed_files / self.total_files) * 100
                if self.total_files > 0
                else 0
            )
            elapsed = time.time() - self.start_time

            # Calculate rate and ETA
            if self.processed_files > 0 and elapsed > 0:
                files_per_second = self.processed_files / elapsed
                remaining_files = self.total_files - self.processed_files
                eta_seconds = (
                    remaining_files / files_per_second if files_per_second > 0 else 0
                )
                eta_str = self._format_time(eta_seconds)
            else:
                files_per_second = 0
                eta_str = "calculating..."

            # Create progress bar
            bar_width = 30
            filled = int(bar_width * percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Format status line
            status_line = (
                f"\r[{bar}] {percent:5.1f}% | "
                f"{self.processed_files}/{self.total_files} | "
                f"✓ {self.successful_files} ✗ {self.failed_files} ⚠ {self.skipped_files} | "
                f"Rate: {files_per_second:.1f}/s | "
                f"ETA: {eta_str}"
            )

            # Add current file if processing
            if self.current_file:
                file_name = Path(self.current_file).name
                if len(file_name) > 30:
                    file_name = file_name[:27] + "..."
                status_line += f" | Current: {file_name}"

            print(status_line, end="", flush=True)

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def finish(self):
        """Mark the end of processing."""
        with self._lock:
            self.end_time = time.time()
            if self.console_output:
                print()  # New line after progress

            # Log summary
            total_time = self.end_time - self.start_time
            self.logger.info(
                f"Processing complete: {self.processed_files} files in {self._format_time(total_time)}"
            )
            self.logger.info(
                f"Summary - Success: {self.successful_files}, "
                f"Failed: {self.failed_files}, "
                f"Skipped: {self.skipped_files}"
            )

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.

        Returns:
            Dictionary with current stats
        """
        with self._lock:
            elapsed = time.time() - self.start_time
            files_per_second = self.processed_files / elapsed if elapsed > 0 else 0

            stats = {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "successful_files": self.successful_files,
                "failed_files": self.failed_files,
                "skipped_files": self.skipped_files,
                "percent_complete": (
                    (self.processed_files / self.total_files * 100)
                    if self.total_files > 0
                    else 0
                ),
                "elapsed_time": elapsed,
                "files_per_second": files_per_second,
                "current_file": self.current_file,
                "errors_count": len(self.errors),
            }

            # Add performance stats
            if self.processing_times:
                stats["avg_processing_time"] = sum(self.processing_times) / len(
                    self.processing_times
                )
                stats["min_processing_time"] = min(self.processing_times)
                stats["max_processing_time"] = max(self.processing_times)

            return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get final processing summary.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total_time = (self.end_time or time.time()) - self.start_time

            summary = {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "successful_files": self.successful_files,
                "failed_files": self.failed_files,
                "skipped_files": self.skipped_files,
                "total_time": total_time,
                "average_speed": (
                    self.processed_files / total_time if total_time > 0 else 0
                ),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": (
                    datetime.fromtimestamp(self.end_time).isoformat()
                    if self.end_time
                    else None
                ),
            }

            # Add performance metrics
            if self.processing_times:
                summary["performance"] = {
                    "avg_processing_time": sum(self.processing_times)
                    / len(self.processing_times),
                    "min_processing_time": min(self.processing_times),
                    "max_processing_time": max(self.processing_times),
                    "total_processing_time": sum(self.processing_times),
                }

            # Add error summary
            if self.errors:
                summary["errors"] = [
                    {"file": file_path, "error": error}
                    for file_path, error in self.errors
                ]

            return summary

    def save_report(self, report_path: str):
        """Save processing report to file.

        Args:
            report_path: Path to save the report
        """
        report = {
            "summary": self.get_summary(),
            "file_statuses": [
                {
                    "file_path": status.file_path,
                    "status": status.status,
                    "timestamp": status.timestamp,
                    "error": status.error,
                    "details": status.details,
                }
                for status in self.file_statuses
            ],
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_path}")

    def generate_error_report(self) -> str:
        """Generate a detailed error report.

        Returns:
            Error report as string
        """
        if not self.errors:
            return "No errors encountered during processing."

        report = ["Error Report", "=" * 40, ""]

        # Group errors by type
        error_types = {}
        for file_path, error in self.errors:
            error_type = (
                type(error).__name__ if isinstance(error, Exception) else "Unknown"
            )
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((file_path, str(error)))

        # Format report
        for error_type, errors in error_types.items():
            report.append(f"\n{error_type} ({len(errors)} occurrences):")
            report.append("-" * 40)
            for file_path, error_msg in errors[:10]:  # Limit to 10 per type
                report.append(f"  File: {file_path}")
                report.append(f"  Error: {error_msg}\n")

            if len(errors) > 10:
                report.append(f"  ... and {len(errors) - 10} more\n")

        return "\n".join(report)


class ConsoleProgressBar:
    """Simple console progress bar for non-file operations."""

    def __init__(self, total: int, description: str = "Progress"):
        """Initialize progress bar.

        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0

    def update(self, increment: int = 1):
        """Update progress.

        Args:
            increment: Number of items to increment
        """
        self.current += increment

        # Throttle updates
        current_time = time.time()
        if current_time - self.last_update < 0.1:
            return

        self.last_update = current_time
        self._display()

    def _display(self):
        """Display progress bar."""
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        print(
            f"\r{self.description}: [{bar}] {percent:5.1f}% | "
            f"{self.current}/{self.total} | "
            f"Rate: {rate:.1f}/s",
            end="",
            flush=True,
        )

    def finish(self):
        """Finish progress bar."""
        self.current = self.total
        self._display()
        print()  # New line


class LogAggregator:
    """Aggregate logs from multiple sources."""

    def __init__(self, output_file: str):
        """Initialize log aggregator.

        Args:
            output_file: Path to aggregated log file
        """
        self.output_file = output_file
        self.logs = []

    def add_log_file(self, log_file: str, source: str):
        """Add a log file to aggregate.

        Args:
            log_file: Path to log file
            source: Source identifier
        """
        try:
            with open(log_file, "r") as f:
                for line in f:
                    self.logs.append(
                        {
                            "source": source,
                            "line": line.strip(),
                            "timestamp": self._extract_timestamp(line),
                        }
                    )
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")

    def _extract_timestamp(self, line: str) -> Optional[float]:
        """Extract timestamp from log line.

        Args:
            line: Log line

        Returns:
            Timestamp or None
        """
        # Common log timestamp patterns
        patterns = [
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",  # ISO format
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",  # ISO with T
        ]

        import re

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    return dt.timestamp()
                except:
                    pass

        return None

    def aggregate(self):
        """Aggregate and sort logs."""
        # Sort by timestamp if available
        self.logs.sort(key=lambda x: x["timestamp"] or 0)

        # Write aggregated logs
        with open(self.output_file, "w") as f:
            for log in self.loops:
                f.write(f"[{log['source']}] {log['line']}\n")
