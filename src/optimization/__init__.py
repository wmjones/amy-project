"""
Performance optimization module for high-volume document processing.
"""

from .performance_optimizer import (
    OptimizedPipeline,
    PerformanceMetrics,
    ResourceMonitor,
    DocumentBatcher,
)

from .monitoring_dashboard import (
    MonitoringDashboard,
    MetricsCollector,
    LiveMetrics,
    ConsoleDashboard,
)

__all__ = [
    "OptimizedPipeline",
    "PerformanceMetrics",
    "ResourceMonitor",
    "DocumentBatcher",
    "MonitoringDashboard",
    "MetricsCollector",
    "LiveMetrics",
    "ConsoleDashboard",
]
