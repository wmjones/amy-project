"""
Real-time monitoring dashboard for pipeline performance tracking.
Provides live metrics and visualization for high-volume document processing.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LiveMetrics:
    """Live performance metrics for real-time monitoring."""

    timestamp: datetime
    files_processed: int
    files_pending: int
    files_failed: int
    current_throughput: float  # Files per minute
    average_processing_time: float
    memory_usage: float
    cpu_usage: float
    active_workers: int
    queue_depth: int
    cache_hits: int
    cache_misses: int
    api_calls: int
    api_errors: int
    last_error: Optional[str] = None


class MetricsCollector:
    """Collect and aggregate performance metrics."""

    def __init__(self, window_size: int = 60):
        """Initialize metrics collector.

        Args:
            window_size: Time window in seconds for rolling metrics
        """
        self.window_size = window_size
        self.metrics_window = deque(maxlen=window_size)
        self.cumulative_metrics = {
            "total_processed": 0,
            "total_failed": 0,
            "total_cache_hits": 0,
            "total_cache_misses": 0,
            "total_api_calls": 0,
            "total_api_errors": 0,
            "processing_times": deque(maxlen=100),
            "start_time": datetime.now(),
        }
        self.lock = threading.Lock()

    def record_processing(
        self,
        processing_time: float,
        success: bool,
        cache_hit: bool,
        api_called: bool,
        api_error: bool = False,
    ):
        """Record a processing event."""
        with self.lock:
            # Update cumulative metrics
            if success:
                self.cumulative_metrics["total_processed"] += 1
            else:
                self.cumulative_metrics["total_failed"] += 1

            if cache_hit:
                self.cumulative_metrics["total_cache_hits"] += 1
            else:
                self.cumulative_metrics["total_cache_misses"] += 1

            if api_called:
                self.cumulative_metrics["total_api_calls"] += 1
                if api_error:
                    self.cumulative_metrics["total_api_errors"] += 1

            self.cumulative_metrics["processing_times"].append(processing_time)

    def get_current_metrics(
        self, system_metrics: Optional[Dict[str, float]] = None
    ) -> LiveMetrics:
        """Get current live metrics."""
        with self.lock:
            now = datetime.now()
            elapsed_time = (now - self.cumulative_metrics["start_time"]).total_seconds()

            # Calculate throughput
            if elapsed_time > 0:
                throughput = (
                    self.cumulative_metrics["total_processed"] / elapsed_time
                ) * 60
            else:
                throughput = 0.0

            # Calculate average processing time
            if self.cumulative_metrics["processing_times"]:
                avg_time = statistics.mean(self.cumulative_metrics["processing_times"])
            else:
                avg_time = 0.0

            # System metrics
            memory_usage = system_metrics.get("memory", 0.0) if system_metrics else 0.0
            cpu_usage = system_metrics.get("cpu", 0.0) if system_metrics else 0.0

            return LiveMetrics(
                timestamp=now,
                files_processed=self.cumulative_metrics["total_processed"],
                files_pending=0,  # Updated by caller
                files_failed=self.cumulative_metrics["total_failed"],
                current_throughput=throughput,
                average_processing_time=avg_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                active_workers=0,  # Updated by caller
                queue_depth=0,  # Updated by caller
                cache_hits=self.cumulative_metrics["total_cache_hits"],
                cache_misses=self.cumulative_metrics["total_cache_misses"],
                api_calls=self.cumulative_metrics["total_api_calls"],
                api_errors=self.cumulative_metrics["total_api_errors"],
            )


class MonitoringDashboard:
    """Real-time monitoring dashboard for pipeline performance."""

    def __init__(self, update_interval: int = 5, metrics_history_size: int = 100):
        """Initialize monitoring dashboard.

        Args:
            update_interval: Update interval in seconds
            metrics_history_size: Number of metric snapshots to keep
        """
        self.update_interval = update_interval
        self.metrics_collector = MetricsCollector()
        self.metrics_history = deque(maxlen=metrics_history_size)

        # Dashboard state
        self.is_running = False
        self.update_thread = None
        self.update_callbacks: List[Callable] = []

        # Performance alerts
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "memory_usage": 85.0,  # 85% memory usage
            "cpu_usage": 90.0,  # 90% CPU usage
            "throughput_min": 10.0,  # Minimum 10 files/minute
            "queue_depth_max": 1000,  # Maximum queue depth
        }
        self.active_alerts = []

    def start(self):
        """Start the monitoring dashboard."""
        if self.is_running:
            return

        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info("Monitoring dashboard started")

    def stop(self):
        """Stop the monitoring dashboard."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=10)

        logger.info("Monitoring dashboard stopped")

    def add_update_callback(self, callback: Callable):
        """Add a callback for metric updates."""
        self.update_callbacks.append(callback)

    def record_event(
        self,
        event_type: str,
        processing_time: float = 0.0,
        success: bool = True,
        cache_hit: bool = False,
        api_called: bool = False,
        api_error: bool = False,
        error_message: Optional[str] = None,
    ):
        """Record a processing event."""
        self.metrics_collector.record_processing(
            processing_time=processing_time,
            success=success,
            cache_hit=cache_hit,
            api_called=api_called,
            api_error=api_error,
        )

        if error_message:
            with self.metrics_collector.lock:
                self.last_error = error_message

    def update_system_metrics(self, system_metrics: Dict[str, float]):
        """Update system resource metrics."""
        # Store for next update cycle
        self._current_system_metrics = system_metrics

    def update_queue_metrics(self, pending: int, active_workers: int, queue_depth: int):
        """Update queue-related metrics."""
        self._queue_metrics = {
            "pending": pending,
            "active_workers": active_workers,
            "queue_depth": queue_depth,
        }

    def _update_loop(self):
        """Main update loop for the dashboard."""
        while self.is_running:
            try:
                # Get current metrics
                metrics = self._collect_current_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Check for alerts
                self._check_alerts(metrics)

                # Notify callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in update callback: {e}")

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in monitoring update loop: {e}")

    def _collect_current_metrics(self) -> LiveMetrics:
        """Collect current metrics from all sources."""
        # Get base metrics
        metrics = self.metrics_collector.get_current_metrics(
            system_metrics=getattr(self, "_current_system_metrics", None)
        )

        # Update queue metrics
        queue_metrics = getattr(self, "_queue_metrics", {})
        metrics.files_pending = queue_metrics.get("pending", 0)
        metrics.active_workers = queue_metrics.get("active_workers", 0)
        metrics.queue_depth = queue_metrics.get("queue_depth", 0)

        return metrics

    def _check_alerts(self, metrics: LiveMetrics):
        """Check for performance alerts."""
        new_alerts = []

        # Check error rate
        total_files = metrics.files_processed + metrics.files_failed
        if total_files > 0:
            error_rate = metrics.files_failed / total_files
            if error_rate > self.alert_thresholds["error_rate"]:
                new_alerts.append(
                    {
                        "type": "error_rate",
                        "severity": "high",
                        "message": f"High error rate: {error_rate*100:.1f}%",
                        "value": error_rate,
                        "threshold": self.alert_thresholds["error_rate"],
                    }
                )

        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            new_alerts.append(
                {
                    "type": "memory_usage",
                    "severity": "high",
                    "message": f"High memory usage: {metrics.memory_usage:.1f}%",
                    "value": metrics.memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"],
                }
            )

        # Check CPU usage
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            new_alerts.append(
                {
                    "type": "cpu_usage",
                    "severity": "medium",
                    "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    "value": metrics.cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage"],
                }
            )

        # Check throughput
        if (
            metrics.files_processed > 10  # Only after processing some files
            and metrics.current_throughput < self.alert_thresholds["throughput_min"]
        ):
            new_alerts.append(
                {
                    "type": "low_throughput",
                    "severity": "medium",
                    "message": f"Low throughput: {metrics.current_throughput:.1f} files/min",
                    "value": metrics.current_throughput,
                    "threshold": self.alert_thresholds["throughput_min"],
                }
            )

        # Check queue depth
        if metrics.queue_depth > self.alert_thresholds["queue_depth_max"]:
            new_alerts.append(
                {
                    "type": "queue_depth",
                    "severity": "medium",
                    "message": f"High queue depth: {metrics.queue_depth}",
                    "value": metrics.queue_depth,
                    "threshold": self.alert_thresholds["queue_depth_max"],
                }
            )

        # Update active alerts
        self.active_alerts = new_alerts

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance."""
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]

        # Calculate trend data
        throughput_trend = self._calculate_trend("current_throughput")
        memory_trend = self._calculate_trend("memory_usage")

        # Estimate completion time
        if latest_metrics.files_pending > 0 and latest_metrics.current_throughput > 0:
            eta_minutes = (
                latest_metrics.files_pending / latest_metrics.current_throughput
            )
            eta = datetime.now() + timedelta(minutes=eta_minutes)
        else:
            eta = None

        summary = {
            "current_state": {
                "files_processed": latest_metrics.files_processed,
                "files_pending": latest_metrics.files_pending,
                "files_failed": latest_metrics.files_failed,
                "throughput": latest_metrics.current_throughput,
                "active_workers": latest_metrics.active_workers,
            },
            "performance": {
                "average_processing_time": latest_metrics.average_processing_time,
                "throughput_trend": throughput_trend,
                "memory_usage": latest_metrics.memory_usage,
                "memory_trend": memory_trend,
                "cpu_usage": latest_metrics.cpu_usage,
            },
            "cache_performance": {
                "hits": latest_metrics.cache_hits,
                "misses": latest_metrics.cache_misses,
                "hit_rate": self._calculate_cache_hit_rate(),
            },
            "api_performance": {
                "total_calls": latest_metrics.api_calls,
                "errors": latest_metrics.api_errors,
                "error_rate": self._calculate_api_error_rate(),
            },
            "estimates": {
                "eta": eta.isoformat() if eta else None,
                "eta_minutes": eta_minutes if "eta_minutes" in locals() else None,
            },
            "alerts": self.active_alerts,
            "last_update": latest_metrics.timestamp.isoformat(),
        }

        return summary

    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric (increasing/decreasing/stable)."""
        if len(self.metrics_history) < 10:
            return "insufficient_data"

        # Get last 10 values
        recent_values = []
        for metrics in list(self.metrics_history)[-10:]:
            value = getattr(metrics, metric_name, 0)
            recent_values.append(value)

        # Calculate trend
        if not recent_values:
            return "no_data"

        avg_first_half = statistics.mean(recent_values[:5])
        avg_second_half = statistics.mean(recent_values[5:])

        change_percentage = (
            ((avg_second_half - avg_first_half) / avg_first_half * 100)
            if avg_first_half
            else 0
        )

        if change_percentage > 5:
            return "increasing"
        elif change_percentage < -5:
            return "decreasing"
        else:
            return "stable"

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.metrics_history:
            return 0.0

        latest = self.metrics_history[-1]
        total_cache_ops = latest.cache_hits + latest.cache_misses

        if total_cache_ops == 0:
            return 0.0

        return latest.cache_hits / total_cache_ops

    def _calculate_api_error_rate(self) -> float:
        """Calculate API error rate."""
        if not self.metrics_history:
            return 0.0

        latest = self.metrics_history[-1]

        if latest.api_calls == 0:
            return 0.0

        return latest.api_errors / latest.api_calls

    def export_metrics(self, output_path: Path):
        """Export metrics history to file."""
        metrics_data = []

        for metrics in self.metrics_history:
            metrics_dict = {
                "timestamp": metrics.timestamp.isoformat(),
                "files_processed": metrics.files_processed,
                "files_pending": metrics.files_pending,
                "files_failed": metrics.files_failed,
                "throughput": metrics.current_throughput,
                "avg_processing_time": metrics.average_processing_time,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "api_calls": metrics.api_calls,
                "api_errors": metrics.api_errors,
            }
            metrics_data.append(metrics_dict)

        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")

    def generate_dashboard_html(self, output_path: Optional[Path] = None) -> str:
        """Generate HTML dashboard view."""
        summary = self.get_performance_summary()

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Pipeline Performance Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .metric-card {{
                    background: #f5f5f5;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #333;
                }}
                .metric-label {{
                    color: #666;
                    margin-bottom: 5px;
                }}
                .alert {{
                    background: #fee;
                    border-left: 4px solid #c33;
                    padding: 10px;
                    margin: 10px 0;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                }}
                .trend-up {{ color: #4a4; }}
                .trend-down {{ color: #a44; }}
                .trend-stable {{ color: #888; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Pipeline Performance Dashboard</h1>
                <p>Last updated: {summary.get('last_update', 'N/A')}</p>

                <div class="grid">
                    <div class="metric-card">
                        <div class="metric-label">Files Processed</div>
                        <div class="metric-value">{summary.get('current_state', {}).get('files_processed', 0)}</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Files Pending</div>
                        <div class="metric-value">{summary.get('current_state', {}).get('files_pending', 0)}</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Throughput</div>
                        <div class="metric-value">{summary.get('current_state', {}).get('throughput', 0):.1f}</div>
                        <div class="metric-label">files/minute</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value">{summary.get('performance', {}).get('memory_usage', 0):.1f}%</div>
                    </div>
                </div>

                <h2>Active Alerts</h2>
                <div id="alerts">
                    {self._generate_alerts_html(summary.get('alerts', []))}
                </div>

                <h2>Performance Trends</h2>
                <div class="grid">
                    <div class="metric-card">
                        <div class="metric-label">Throughput Trend</div>
                        <div class="metric-value {self._get_trend_class(summary.get('performance', {}).get('throughput_trend', ''))}">
                            {summary.get('performance', {}).get('throughput_trend', 'N/A')}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Cache Hit Rate</div>
                        <div class="metric-value">
                            {summary.get('cache_performance', {}).get('hit_rate', 0)*100:.1f}%
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">API Error Rate</div>
                        <div class="metric-value">
                            {summary.get('api_performance', {}).get('error_rate', 0)*100:.1f}%
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">ETA</div>
                        <div class="metric-value">
                            {summary.get('estimates', {}).get('eta_minutes', 'N/A')} min
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        if output_path:
            with open(output_path, "w") as f:
                f.write(html)

        return html

    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts."""
        if not alerts:
            return "<p>No active alerts</p>"

        html_parts = []
        for alert in alerts:
            html_parts.append(
                f'<div class="alert">'
                f'<strong>{alert["type"].replace("_", " ").title()}</strong>: '
                f'{alert["message"]}'
                f"</div>"
            )

        return "\n".join(html_parts)

    def _get_trend_class(self, trend: str) -> str:
        """Get CSS class for trend display."""
        trend_classes = {
            "increasing": "trend-up",
            "decreasing": "trend-down",
            "stable": "trend-stable",
        }
        return trend_classes.get(trend, "")


# Console-based dashboard display
class ConsoleDashboard:
    """Console-based dashboard display."""

    @staticmethod
    def display_metrics(metrics: LiveMetrics):
        """Display metrics in console."""
        print("\033[H\033[J")  # Clear screen
        print("=" * 50)
        print("Pipeline Performance Dashboard")
        print("=" * 50)
        print(f"Time: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"Files Processed: {metrics.files_processed}")
        print(f"Files Pending: {metrics.files_pending}")
        print(f"Files Failed: {metrics.files_failed}")
        print(f"Throughput: {metrics.current_throughput:.1f} files/min")
        print(f"Avg Processing Time: {metrics.average_processing_time:.1f}s")
        print()
        print(f"Memory Usage: {metrics.memory_usage:.1f}%")
        print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
        print(f"Active Workers: {metrics.active_workers}")
        print(f"Queue Depth: {metrics.queue_depth}")
        print()
        print(f"Cache Hits: {metrics.cache_hits}")
        print(f"Cache Misses: {metrics.cache_misses}")
        print(f"API Calls: {metrics.api_calls}")
        print(f"API Errors: {metrics.api_errors}")

        if metrics.last_error:
            print()
            print(f"Last Error: {metrics.last_error}")

        print("=" * 50)
