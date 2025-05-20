"""
Statistics collection and analysis for file organization.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics
from pathlib import Path
from collections import defaultdict


@dataclass
class FileStatistics:
    """Statistics for a single file."""

    file_path: str
    file_type: str
    file_size: int
    processing_time: float
    operation: str  # 'move', 'copy', 'skip'
    success: bool
    error: Optional[str] = None
    metadata_extracted: bool = False
    confidence_score: Optional[float] = None
    target_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class StatisticsCollector:
    """Collect and analyze statistics for file organization."""

    def __init__(self):
        """Initialize statistics collector."""
        self.start_time = time.time()
        self.end_time = None
        self.file_stats = []
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.file_type_stats = defaultdict(
            lambda: {"count": 0, "total_size": 0, "total_time": 0, "errors": 0}
        )

        # Performance metrics
        self.processing_times = []
        self.file_sizes = []
        self.confidence_scores = []

        # Throughput tracking
        self.bytes_processed = 0
        self.files_processed = 0

    def add_file_stat(self, file_stat: FileStatistics):
        """Add statistics for a processed file.

        Args:
            file_stat: FileStatistics object
        """
        self.file_stats.append(file_stat)

        # Update counters
        self.files_processed += 1
        self.operation_counts[file_stat.operation] += 1

        if file_stat.success:
            self.bytes_processed += file_stat.file_size
            self.processing_times.append(file_stat.processing_time)
            self.file_sizes.append(file_stat.file_size)

            if file_stat.confidence_score is not None:
                self.confidence_scores.append(file_stat.confidence_score)
        else:
            error_type = (
                type(file_stat.error).__name__ if file_stat.error else "Unknown"
            )
            self.error_counts[error_type] += 1

        # Update file type statistics
        file_type = file_stat.file_type.lower()
        type_stats = self.file_type_stats[file_type]
        type_stats["count"] += 1
        type_stats["total_size"] += file_stat.file_size
        type_stats["total_time"] += file_stat.processing_time
        if not file_stat.success:
            type_stats["errors"] += 1

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics.

        Returns:
            Dictionary with calculated statistics
        """
        total_time = (self.end_time or time.time()) - self.start_time

        stats = {
            "summary": {
                "total_files": len(self.file_stats),
                "successful_files": sum(1 for s in self.file_stats if s.success),
                "failed_files": sum(1 for s in self.file_stats if not s.success),
                "total_bytes": self.bytes_processed,
                "total_time": total_time,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": (
                    datetime.fromtimestamp(self.end_time).isoformat()
                    if self.end_time
                    else None
                ),
            }
        }

        # Performance metrics
        if self.processing_times:
            stats["performance"] = {
                "avg_processing_time": statistics.mean(self.processing_times),
                "median_processing_time": statistics.median(self.processing_times),
                "min_processing_time": min(self.processing_times),
                "max_processing_time": max(self.processing_times),
                "std_dev_processing_time": (
                    statistics.stdev(self.processing_times)
                    if len(self.processing_times) > 1
                    else 0
                ),
                "files_per_second": (
                    self.files_processed / total_time if total_time > 0 else 0
                ),
                "bytes_per_second": (
                    self.bytes_processed / total_time if total_time > 0 else 0
                ),
            }

        # File size analysis
        if self.file_sizes:
            stats["file_sizes"] = {
                "avg_size": statistics.mean(self.file_sizes),
                "median_size": statistics.median(self.file_sizes),
                "min_size": min(self.file_sizes),
                "max_size": max(self.file_sizes),
                "total_size": sum(self.file_sizes),
            }

        # Confidence score analysis
        if self.confidence_scores:
            stats["confidence_scores"] = {
                "avg_confidence": statistics.mean(self.confidence_scores),
                "median_confidence": statistics.median(self.confidence_scores),
                "min_confidence": min(self.confidence_scores),
                "max_confidence": max(self.confidence_scores),
            }

        # Operation breakdown
        stats["operations"] = dict(self.operation_counts)

        # Error analysis
        stats["errors"] = dict(self.error_counts)

        # File type breakdown
        stats["file_types"] = {}
        for file_type, type_stats in self.file_type_stats.items():
            stats["file_types"][file_type] = {
                "count": type_stats["count"],
                "total_size": type_stats["total_size"],
                "avg_size": (
                    type_stats["total_size"] / type_stats["count"]
                    if type_stats["count"] > 0
                    else 0
                ),
                "total_time": type_stats["total_time"],
                "avg_time": (
                    type_stats["total_time"] / type_stats["count"]
                    if type_stats["count"] > 0
                    else 0
                ),
                "error_rate": (
                    type_stats["errors"] / type_stats["count"]
                    if type_stats["count"] > 0
                    else 0
                ),
            }

        return stats

    def generate_report(self) -> str:
        """Generate a human-readable statistics report.

        Returns:
            Report as string
        """
        stats = self.calculate_statistics()
        report = []

        # Header
        report.append("File Organization Statistics Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        summary = stats["summary"]
        report.append("Summary")
        report.append("-" * 20)
        report.append(f"Total Files: {summary['total_files']}")
        report.append(f"Successful: {summary['successful_files']}")
        report.append(f"Failed: {summary['failed_files']}")
        report.append(f"Total Size: {self._format_bytes(summary['total_bytes'])}")
        report.append(f"Total Time: {self._format_duration(summary['total_time'])}")
        report.append("")

        # Performance
        if "performance" in stats:
            perf = stats["performance"]
            report.append("Performance")
            report.append("-" * 20)
            report.append(f"Files/second: {perf['files_per_second']:.2f}")
            report.append(
                f"Throughput: {self._format_bytes(perf['bytes_per_second'])}/s"
            )
            report.append(f"Avg Processing Time: {perf['avg_processing_time']:.3f}s")
            report.append(
                f"Median Processing Time: {perf['median_processing_time']:.3f}s"
            )
            report.append("")

        # File Types
        if "file_types" in stats:
            report.append("File Types")
            report.append("-" * 20)
            for file_type, type_stats in sorted(
                stats["file_types"].items(), key=lambda x: x[1]["count"], reverse=True
            ):
                report.append(f"{file_type}:")
                report.append(f"  Count: {type_stats['count']}")
                report.append(
                    f"  Total Size: {self._format_bytes(type_stats['total_size'])}"
                )
                report.append(f"  Avg Time: {type_stats['avg_time']:.3f}s")
                if type_stats["error_rate"] > 0:
                    report.append(f"  Error Rate: {type_stats['error_rate']*100:.1f}%")
            report.append("")

        # Errors
        if stats.get("errors"):
            report.append("Errors")
            report.append("-" * 20)
            for error_type, count in sorted(
                stats["errors"].items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"{error_type}: {count}")
            report.append("")

        # Operations
        if stats.get("operations"):
            report.append("Operations")
            report.append("-" * 20)
            for operation, count in stats["operations"].items():
                report.append(f"{operation}: {count}")
            report.append("")

        return "\n".join(report)

    def save_statistics(self, output_path: str):
        """Save statistics to file.

        Args:
            output_path: Path to save statistics
        """
        stats = self.calculate_statistics()

        # Add detailed file statistics
        stats["files"] = [
            {
                "file_path": fs.file_path,
                "file_type": fs.file_type,
                "file_size": fs.file_size,
                "processing_time": fs.processing_time,
                "operation": fs.operation,
                "success": fs.success,
                "error": fs.error,
                "confidence_score": fs.confidence_score,
                "target_path": fs.target_path,
                "timestamp": fs.timestamp,
            }
            for fs in self.file_stats
        ]

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format.

        Args:
            bytes_value: Number of bytes

        Returns:
            Formatted string
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def finish(self):
        """Mark the end of statistics collection."""
        self.end_time = time.time()

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error summaries
        """
        error_files = defaultdict(list)

        for stat in self.file_stats:
            if not stat.success and stat.error:
                error_type = (
                    type(stat.error).__name__
                    if isinstance(stat.error, Exception)
                    else "Error"
                )
                error_files[stat.error].append(stat.file_path)

        # Sort by frequency
        sorted_errors = sorted(
            error_files.items(), key=lambda x: len(x[1]), reverse=True
        )[:limit]

        return [
            {
                "error": error,
                "count": len(files),
                "example_files": files[:3],  # Show first 3 examples
            }
            for error, files in sorted_errors
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.

        Returns:
            Performance metrics summary
        """
        if not self.processing_times:
            return {}

        # Calculate percentiles
        sorted_times = sorted(self.processing_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p90 = sorted_times[int(len(sorted_times) * 0.9)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        return {
            "avg_processing_time": statistics.mean(self.processing_times),
            "median_processing_time": p50,
            "p90_processing_time": p90,
            "p99_processing_time": p99,
            "slowest_files": self._get_slowest_files(5),
            "fastest_files": self._get_fastest_files(5),
        }

    def _get_slowest_files(self, limit: int) -> List[Dict[str, Any]]:
        """Get slowest processed files.

        Args:
            limit: Number of files to return

        Returns:
            List of file information
        """
        sorted_stats = sorted(
            self.file_stats, key=lambda x: x.processing_time, reverse=True
        )[:limit]

        return [
            {
                "file": Path(stat.file_path).name,
                "time": stat.processing_time,
                "size": stat.file_size,
                "type": stat.file_type,
            }
            for stat in sorted_stats
        ]

    def _get_fastest_files(self, limit: int) -> List[Dict[str, Any]]:
        """Get fastest processed files.

        Args:
            limit: Number of files to return

        Returns:
            List of file information
        """
        sorted_stats = sorted(self.file_stats, key=lambda x: x.processing_time)[:limit]

        return [
            {
                "file": Path(stat.file_path).name,
                "time": stat.processing_time,
                "size": stat.file_size,
                "type": stat.file_type,
            }
            for stat in sorted_stats
        ]
