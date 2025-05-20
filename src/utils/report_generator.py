"""
Report generation system for the file organization process.
Creates detailed reports on processing results, statistics, and performance.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import html

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports on file organization process."""

    def __init__(self, start_time: Optional[float] = None):
        """
        Initialize report generator.

        Args:
            start_time: Start time of the organization process
        """
        self.start_time = start_time or time.time()
        self.operation_stats = defaultdict(int)
        self.file_movements: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.rule_usage = defaultdict(int)
        self.type_distribution = defaultdict(int)
        self.size_distribution = defaultdict(int)
        self.date_distribution = defaultdict(int)

    def record_file_movement(
        self, source: Path, destination: Path, metadata: Dict[str, Any], rule_name: str
    ):
        """Record a file movement for reporting."""
        movement = {
            "source": str(source),
            "destination": str(destination),
            "size": source.stat().st_size if source.exists() else 0,
            "type": metadata.get("document_type", "unknown"),
            "rule": rule_name,
            "timestamp": datetime.now().isoformat(),
        }
        self.file_movements.append(movement)

        # Update statistics
        self.rule_usage[rule_name] += 1
        self.type_distribution[movement["type"]] += 1
        self._update_size_distribution(movement["size"])

        # Extract date for date distribution
        if "dates" in metadata and "document_date" in metadata["dates"]:
            date_str = metadata["dates"]["document_date"]
            self._update_date_distribution(date_str)

    def record_error(
        self, file_path: Path, error: Exception, context: str = "processing"
    ):
        """Record an error for reporting."""
        error_info = {
            "file": str(file_path),
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)

    def record_operation(self, operation_type: str):
        """Record an operation (copy, move, skip, etc.)."""
        self.operation_stats[operation_type] += 1

    def _update_size_distribution(self, size_bytes: int):
        """Update size distribution statistics."""
        if size_bytes < 100 * 1024:  # < 100KB
            self.size_distribution["< 100KB"] += 1
        elif size_bytes < 1 * 1024 * 1024:  # < 1MB
            self.size_distribution["100KB - 1MB"] += 1
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            self.size_distribution["1MB - 10MB"] += 1
        elif size_bytes < 100 * 1024 * 1024:  # < 100MB
            self.size_distribution["10MB - 100MB"] += 1
        else:
            self.size_distribution["> 100MB"] += 1

    def _update_date_distribution(self, date_str: str):
        """Update date distribution statistics."""
        try:
            date = datetime.fromisoformat(date_str)
            year_month = date.strftime("%Y-%m")
            self.date_distribution[year_month] += 1
        except:
            self.date_distribution["unknown"] += 1

    def generate_summary_report(
        self,
        output_format: str = "text",
        total_files: int = 0,
        processed_files: int = 0,
        successful_files: int = 0,
        failed_files: int = 0,
        skipped_files: int = 0,
    ) -> str:
        """
        Generate a summary report of the organization process.

        Args:
            output_format: Format for the report ('text', 'html', 'json')
            total_files: Total number of files to process
            processed_files: Number of files processed
            successful_files: Number of files successfully organized
            failed_files: Number of files that failed
            skipped_files: Number of files skipped

        Returns:
            Formatted report string
        """
        elapsed_time = time.time() - self.start_time

        # Compile statistics
        stats = {
            "summary": {
                "total_files": total_files,
                "processed_files": processed_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "skipped_files": skipped_files,
                "success_rate": (
                    (successful_files / processed_files * 100)
                    if processed_files > 0
                    else 0
                ),
            },
            "performance": {
                "elapsed_time": elapsed_time,
                "elapsed_time_formatted": str(timedelta(seconds=int(elapsed_time))),
                "files_per_second": (
                    processed_files / elapsed_time if elapsed_time > 0 else 0
                ),
            },
            "operations": dict(self.operation_stats),
            "rule_usage": dict(self.rule_usage),
            "type_distribution": dict(self.type_distribution),
            "size_distribution": dict(self.size_distribution),
            "date_distribution": dict(self.date_distribution),
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # Include first 10 errors
        }

        # Generate report in requested format
        if output_format == "text":
            return self._generate_text_report(stats)
        elif output_format == "html":
            return self._generate_html_report(stats)
        elif output_format == "json":
            return json.dumps(stats, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_text_report(self, stats: Dict[str, Any]) -> str:
        """Generate a text format report."""
        report = ["===== File Organization Report ====="]
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary section
        summary = stats["summary"]
        report.append("PROCESSING SUMMARY")
        report.append("-" * 18)
        report.append(f"Total files:      {summary['total_files']:,}")
        if summary["total_files"] > 0:
            processed_pct = summary["processed_files"] / summary["total_files"] * 100
        else:
            processed_pct = 0
        report.append(
            f"Processed:        {summary['processed_files']:,} ({processed_pct:.1f}%)"
        )
        report.append(
            f"Successful:       {summary['successful_files']:,} ({summary['success_rate']:.1f}%)"
        )
        report.append(f"Failed:           {summary['failed_files']:,}")
        report.append(f"Skipped:          {summary['skipped_files']:,}")
        report.append("")

        # Performance section
        perf = stats["performance"]
        report.append("PERFORMANCE")
        report.append("-" * 11)
        report.append(f"Total time:       {perf['elapsed_time_formatted']}")
        report.append(f"Processing rate:  {perf['files_per_second']:.2f} files/second")
        report.append("")

        # Operations breakdown
        if stats["operations"]:
            report.append("OPERATIONS")
            report.append("-" * 10)
            for op, count in sorted(stats["operations"].items()):
                report.append(f"{op:12s}  {count:,}")
            report.append("")

        # Rule usage
        if stats["rule_usage"]:
            report.append("RULE USAGE")
            report.append("-" * 10)
            for rule, count in sorted(
                stats["rule_usage"].items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"{rule[:30]:30s}  {count:,}")
            report.append("")

        # Type distribution
        if stats["type_distribution"]:
            report.append("FILE TYPE DISTRIBUTION")
            report.append("-" * 21)
            for file_type, count in sorted(
                stats["type_distribution"].items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"{file_type:20s}  {count:,}")
            report.append("")

        # Size distribution
        if stats["size_distribution"]:
            report.append("FILE SIZE DISTRIBUTION")
            report.append("-" * 21)
            size_order = [
                "< 100KB",
                "100KB - 1MB",
                "1MB - 10MB",
                "10MB - 100MB",
                "> 100MB",
            ]
            for size_range in size_order:
                if size_range in stats["size_distribution"]:
                    count = stats["size_distribution"][size_range]
                    report.append(f"{size_range:15s}  {count:,}")
            report.append("")

        # Date distribution (top 10)
        if stats["date_distribution"]:
            report.append("DATE DISTRIBUTION (Top 10)")
            report.append("-" * 25)
            sorted_dates = sorted(
                stats["date_distribution"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            for date, count in sorted_dates:
                report.append(f"{date:10s}  {count:,}")
            report.append("")

        # Errors
        if stats["error_count"] > 0:
            report.append(f"ERRORS ({stats['error_count']} total)")
            report.append("-" * 6)
            for error in stats["errors"]:
                report.append(f"{error['context']} - {error['file']}: {error['error']}")
            if stats["error_count"] > len(stats["errors"]):
                report.append(
                    f"... and {stats['error_count'] - len(stats['errors'])} more errors"
                )
            report.append("")

        return "\n".join(report)

    def _generate_html_report(self, stats: Dict[str, Any]) -> str:
        """Generate an HTML format report."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>File Organization Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
            "th { background-color: #f2f2f2; }",
            ".summary-box { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }",
            ".error-box { background-color: #fee; padding: 10px; border-left: 3px solid #d00; margin: 10px 0; }",
            ".success-rate { color: #0a0; font-weight: bold; }",
            ".fail-rate { color: #d00; font-weight: bold; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>File Organization Report</h1>",
            f'<p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]

        # Summary section
        summary = stats["summary"]
        html_parts.extend(
            [
                '<div class="summary-box">',
                "<h2>Processing Summary</h2>",
                "<table>",
                f'<tr><td>Total files:</td><td>{summary["total_files"]:,}</td></tr>',
                f'<tr><td>Processed:</td><td>{summary["processed_files"]:,} ({(summary["processed_files"]/summary["total_files"]*100 if summary["total_files"] > 0 else 0.0):.1f}%)</td></tr>',
                f'<tr><td>Successful:</td><td class="success-rate">{summary["successful_files"]:,} ({summary["success_rate"]:.1f}%)</td></tr>',
                f'<tr><td>Failed:</td><td class="fail-rate">{summary["failed_files"]:,}</td></tr>',
                f'<tr><td>Skipped:</td><td>{summary["skipped_files"]:,}</td></tr>',
                "</table>",
                "</div>",
            ]
        )

        # Performance section
        perf = stats["performance"]
        html_parts.extend(
            [
                '<div class="summary-box">',
                "<h2>Performance</h2>",
                "<table>",
                f'<tr><td>Total time:</td><td>{perf["elapsed_time_formatted"]}</td></tr>',
                f'<tr><td>Processing rate:</td><td>{perf["files_per_second"]:.2f} files/second</td></tr>',
                "</table>",
                "</div>",
            ]
        )

        # Operations breakdown
        if stats["operations"]:
            html_parts.extend(
                [
                    "<h2>Operations</h2>",
                    "<table>",
                    "<tr><th>Operation</th><th>Count</th></tr>",
                ]
            )
            for op, count in sorted(stats["operations"].items()):
                html_parts.append(
                    f"<tr><td>{html.escape(op)}</td><td>{count:,}</td></tr>"
                )
            html_parts.append("</table>")

        # Rule usage
        if stats["rule_usage"]:
            html_parts.extend(
                [
                    "<h2>Rule Usage</h2>",
                    "<table>",
                    "<tr><th>Rule</th><th>Count</th></tr>",
                ]
            )
            for rule, count in sorted(
                stats["rule_usage"].items(), key=lambda x: x[1], reverse=True
            ):
                html_parts.append(
                    f"<tr><td>{html.escape(rule)}</td><td>{count:,}</td></tr>"
                )
            html_parts.append("</table>")

        # Charts placeholders (could be enhanced with JavaScript)
        html_parts.extend(
            [
                "<h2>Distributions</h2>",
                '<div style="display: flex; gap: 20px;">',
                '<div style="flex: 1;">',
                "<h3>File Types</h3>",
                "<table>",
                "<tr><th>Type</th><th>Count</th></tr>",
            ]
        )

        for file_type, count in sorted(
            stats["type_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            html_parts.append(
                f"<tr><td>{html.escape(file_type)}</td><td>{count:,}</td></tr>"
            )
        html_parts.extend(["</table>", "</div>"])

        # Size distribution
        html_parts.extend(
            [
                '<div style="flex: 1;">',
                "<h3>File Sizes</h3>",
                "<table>",
                "<tr><th>Size Range</th><th>Count</th></tr>",
            ]
        )

        size_order = ["< 100KB", "100KB - 1MB", "1MB - 10MB", "10MB - 100MB", "> 100MB"]
        for size_range in size_order:
            if size_range in stats["size_distribution"]:
                count = stats["size_distribution"][size_range]
                html_parts.append(
                    f"<tr><td>{html.escape(size_range)}</td><td>{count:,}</td></tr>"
                )

        html_parts.extend(["</table>", "</div>", "</div>"])

        # Errors
        if stats["error_count"] > 0:
            html_parts.extend(
                [
                    f'<h2>Errors ({stats["error_count"]} total)</h2>',
                    '<div class="error-box">',
                ]
            )
            for error in stats["errors"][:10]:
                html_parts.append(
                    f'<p><strong>{html.escape(error["context"])}</strong> - '
                    f'{html.escape(error["file"])}: {html.escape(error["error"])}</p>'
                )
            if stats["error_count"] > len(stats["errors"]):
                html_parts.append(
                    f'<p>... and {stats["error_count"] - len(stats["errors"])} more errors</p>'
                )
            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def generate_movement_report(self, output_path: Path, format: str = "json"):
        """
        Generate a detailed report of all file movements.

        Args:
            output_path: Path to save the report
            format: Report format ('json', 'csv')
        """
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.file_movements, f, indent=2)
        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                if self.file_movements:
                    writer = csv.DictWriter(f, fieldnames=self.file_movements[0].keys())
                    writer.writeheader()
                    writer.writerows(self.file_movements)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Movement report saved to {output_path}")

    def generate_error_report(self, output_path: Path):
        """Generate a detailed error report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_errors": len(self.errors),
            "errors_by_type": self._group_errors_by_type(),
            "errors_by_context": self._group_errors_by_context(),
            "all_errors": self.errors,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Error report saved to {output_path}")

    def _group_errors_by_type(self) -> Dict[str, int]:
        """Group errors by exception type."""
        type_counts = defaultdict(int)
        for error in self.errors:
            type_counts[error["type"]] += 1
        return dict(type_counts)

    def _group_errors_by_context(self) -> Dict[str, int]:
        """Group errors by context."""
        context_counts = defaultdict(int)
        for error in self.errors:
            context_counts[error["context"]] += 1
        return dict(context_counts)

    def generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data suitable for creating visualizations."""
        return {
            "file_types": dict(self.type_distribution),
            "file_sizes": dict(self.size_distribution),
            "dates": dict(self.date_distribution),
            "rules": dict(self.rule_usage),
            "operations": dict(self.operation_stats),
        }

    def save_all_reports(self, output_dir: Path):
        """Save all report types to the specified directory."""
        output_dir.mkdir(exist_ok=True)

        # Basic reports
        summary_text = self.generate_summary_report("text")
        with open(output_dir / "summary.txt", "w") as f:
            f.write(summary_text)

        summary_html = self.generate_summary_report("html")
        with open(output_dir / "summary.html", "w") as f:
            f.write(summary_html)

        summary_json = self.generate_summary_report("json")
        with open(output_dir / "summary.json", "w") as f:
            f.write(summary_json)

        # Detailed reports
        if self.file_movements:
            self.generate_movement_report(output_dir / "movements.json")
            self.generate_movement_report(output_dir / "movements.csv", format="csv")

        if self.errors:
            self.generate_error_report(output_dir / "errors.json")

        # Visualization data
        viz_data = self.generate_visualization_data()
        with open(output_dir / "visualization_data.json", "w") as f:
            json.dump(viz_data, f, indent=2)

        logger.info(f"All reports saved to {output_dir}")
