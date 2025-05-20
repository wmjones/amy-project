"""
Test suite for performance optimization and monitoring components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time
import threading
from datetime import datetime
import json

from src.optimization.performance_optimizer import (
    OptimizedPipeline,
    PerformanceMetrics,
    ResourceMonitor,
    DocumentBatcher,
)
from src.optimization.monitoring_dashboard import (
    MonitoringDashboard,
    MetricsCollector,
    LiveMetrics,
)


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality."""

    def test_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        self.assertIsNotNone(monitor.metrics)
        self.assertFalse(monitor.monitoring)
        self.assertIsNone(monitor.monitor_thread)

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    def test_monitoring_cycle(self, mock_cpu, mock_memory):
        """Test monitoring cycle."""
        # Mock system metrics
        mock_memory.return_value = Mock(percent=65.5)
        mock_cpu.return_value = 45.2

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Let it run for a bit
        time.sleep(2)

        monitor.stop_monitoring()

        # Check metrics were collected
        self.assertGreater(len(monitor.metrics["memory"]), 0)
        self.assertGreater(len(monitor.metrics["cpu"]), 0)
        self.assertIn(65.5, monitor.metrics["memory"])
        self.assertIn(45.2, monitor.metrics["cpu"])

    def test_peak_metrics(self):
        """Test peak metrics calculation."""
        monitor = ResourceMonitor()

        # Inject test data
        monitor.metrics["memory"] = [50.0, 75.0, 60.0]
        monitor.metrics["cpu"] = [30.0, 90.0, 45.0]

        peak_metrics = monitor.get_peak_metrics()

        self.assertEqual(peak_metrics["peak_memory"], 75.0)
        self.assertEqual(peak_metrics["peak_cpu"], 90.0)
        self.assertAlmostEqual(peak_metrics["average_memory"], 61.67, places=1)
        self.assertAlmostEqual(peak_metrics["average_cpu"], 55.0, places=1)


class TestDocumentBatcher(unittest.TestCase):
    """Test document batching functionality."""

    def setUp(self):
        """Set up test files."""
        self.test_dir = Path("./test_batch_data")
        self.test_dir.mkdir(exist_ok=True)

        # Create test files with different sizes
        self.test_files = []
        for i in range(10):
            file_path = self.test_dir / f"test_{i}.txt"
            # Create files with varying sizes (1KB to 10MB)
            content = "x" * (1024 * (i + 1) * 100)  # Simulated content
            file_path.write_text(content)
            self.test_files.append(file_path)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_batch_creation(self):
        """Test batch creation logic."""
        batcher = DocumentBatcher(batch_size=3, max_batch_memory=500)

        batches = batcher.create_batches(self.test_files[:6])

        # Should create multiple batches
        self.assertGreater(len(batches), 1)

        # Check batch sizes
        for batch in batches:
            self.assertLessEqual(len(batch), 3)

    def test_size_prioritization(self):
        """Test batch creation with size prioritization."""
        batcher = DocumentBatcher(batch_size=5)

        batches = batcher.create_batches(self.test_files, prioritize_by_size=True)

        # First batch should have smaller files
        first_batch = batches[0]
        first_batch_sizes = [f.stat().st_size for f in first_batch]

        # Should be sorted by size
        self.assertEqual(first_batch_sizes, sorted(first_batch_sizes))


class TestOptimizedPipeline(unittest.TestCase):
    """Test optimized pipeline functionality."""

    def setUp(self):
        """Set up test components."""
        self.mock_ocr = Mock()
        self.mock_ai = Mock()
        self.mock_claude = Mock()

        self.pipeline = OptimizedPipeline(
            ocr_processor=self.mock_ocr,
            ai_summarizer=self.mock_ai,
            claude_client=self.mock_claude,
            max_workers=4,
            use_multiprocessing=False,  # Easier to test
            cache_enabled=False,
            batch_size=5,
        )

        # Create test files
        self.test_files = []
        for i in range(10):
            self.test_files.append(Path(f"test_{i}.txt"))

    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.max_workers, 4)
        self.assertEqual(self.pipeline.batch_size, 5)
        self.assertIsNotNone(self.pipeline.resource_monitor)

    def test_optimization_config(self):
        """Test optimization configuration."""
        # Test different levels
        speed_config = self.pipeline._get_optimization_config("speed")
        self.assertEqual(speed_config["batch_size"], 20)
        self.assertTrue(speed_config["parallel_ocr"])

        balanced_config = self.pipeline._get_optimization_config("balanced")
        self.assertEqual(balanced_config["batch_size"], 10)
        self.assertFalse(balanced_config["parallel_ai"])

        quality_config = self.pipeline._get_optimization_config("quality")
        self.assertEqual(quality_config["batch_size"], 5)
        self.assertFalse(quality_config["use_multiprocessing"])

    @patch("src.optimization.performance_optimizer.ProcessPoolExecutor")
    def test_batch_processing(self, mock_executor):
        """Test batch processing logic."""
        # Mock OCR processing
        ocr_result = Mock()
        ocr_result.text = "Test text"
        ocr_result.confidence = 0.9

        self.pipeline._process_ocr = Mock(return_value=ocr_result)
        self.pipeline._process_ai_with_rate_limit = Mock()

        # Process a batch
        config = self.pipeline._get_optimization_config("balanced")
        batch = self.test_files[:3]

        results = self.pipeline._process_batch(batch, config, 0, 1)

        # Check that OCR was called for each file
        self.assertEqual(self.pipeline._process_ocr.call_count, 3)

    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Set low rate limit for testing
        self.pipeline.api_rate_limit = 2
        self.pipeline.api_call_times = []

        # Make rapid calls
        start_time = time.time()

        for i in range(3):
            self.pipeline._enforce_rate_limit()

        # Should have taken some time due to rate limiting
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.5)  # Should wait

    def test_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Create mock results
        from src.metadata_extraction.ocr_ai_pipeline import PipelineResult

        results = []
        for i in range(5):
            result = Mock(spec=PipelineResult)
            result.success = i < 4  # 1 failure
            result.processing_time = 2.0
            results.append(result)

        # Mock metrics data
        self.pipeline.metrics["ocr_times"] = [1.0, 1.2, 0.8, 1.1, 0.9]
        self.pipeline.metrics["ai_times"] = [0.8, 0.9, 0.7, 0.9, 0.8]
        self.pipeline.metrics["cache_hits"] = 2
        self.pipeline.metrics["cache_misses"] = 3

        # Mock resource monitor
        self.pipeline.resource_monitor.get_peak_metrics = Mock(
            return_value={"peak_memory": 75.5, "average_cpu": 45.2}
        )

        metrics = self.pipeline._calculate_metrics(results, 10.0, 5)

        self.assertEqual(metrics.total_files, 5)
        self.assertEqual(metrics.processed_files, 4)
        self.assertEqual(metrics.failed_files, 1)
        self.assertEqual(metrics.total_time, 10.0)
        self.assertAlmostEqual(metrics.cache_hit_rate, 0.4)
        self.assertAlmostEqual(metrics.error_rate, 0.2)

    def test_bottleneck_analysis(self):
        """Test bottleneck analysis."""
        metrics = PerformanceMetrics(
            total_files=100,
            processed_files=95,
            failed_files=5,
            total_time=200.0,
            average_time_per_file=2.0,
            peak_memory_usage=85.0,
            cpu_utilization=40.0,
            throughput=0.5,
            ocr_time=1.4,
            ai_time=0.5,
            io_time=0.1,
            cache_hit_rate=0.1,
            error_rate=0.05,
            timestamp=datetime.now(),
        )

        analysis = self.pipeline.analyze_bottlenecks(metrics)

        # Should identify OCR as bottleneck
        self.assertIn("OCR processing", analysis["bottlenecks"])
        self.assertIn("Memory usage", analysis["bottlenecks"])

        # Should have recommendations
        self.assertGreater(len(analysis["recommendations"]), 0)

        # Performance grade should be low due to low throughput
        self.assertEqual(analysis["performance_grade"], "D")


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection functionality."""

    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(window_size=30)
        self.assertEqual(collector.window_size, 30)
        self.assertEqual(collector.cumulative_metrics["total_processed"], 0)
        self.assertIsNotNone(collector.lock)

    def test_record_processing(self):
        """Test recording processing events."""
        collector = MetricsCollector()

        # Record successful processing
        collector.record_processing(
            processing_time=2.5,
            success=True,
            cache_hit=True,
            api_called=True,
            api_error=False,
        )

        # Check metrics
        self.assertEqual(collector.cumulative_metrics["total_processed"], 1)
        self.assertEqual(collector.cumulative_metrics["total_cache_hits"], 1)
        self.assertEqual(collector.cumulative_metrics["total_api_calls"], 1)
        self.assertEqual(collector.cumulative_metrics["total_api_errors"], 0)

        # Record failed processing
        collector.record_processing(
            processing_time=0.5,
            success=False,
            cache_hit=False,
            api_called=True,
            api_error=True,
        )

        self.assertEqual(collector.cumulative_metrics["total_failed"], 1)
        self.assertEqual(collector.cumulative_metrics["total_api_errors"], 1)

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        collector = MetricsCollector()

        # Add some data
        for i in range(5):
            collector.record_processing(
                processing_time=1.0 + i * 0.1,
                success=True,
                cache_hit=i % 2 == 0,
                api_called=True,
                api_error=False,
            )

        # Get metrics
        metrics = collector.get_current_metrics()

        self.assertIsInstance(metrics, LiveMetrics)
        self.assertEqual(metrics.files_processed, 5)
        self.assertEqual(metrics.cache_hits, 3)
        self.assertEqual(metrics.cache_misses, 2)


class TestMonitoringDashboard(unittest.TestCase):
    """Test monitoring dashboard functionality."""

    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = MonitoringDashboard(update_interval=5)
        self.assertEqual(dashboard.update_interval, 5)
        self.assertIsNotNone(dashboard.metrics_collector)
        self.assertFalse(dashboard.is_running)

    def test_alert_detection(self):
        """Test performance alert detection."""
        dashboard = MonitoringDashboard()

        # Create metrics with issues
        metrics = LiveMetrics(
            timestamp=datetime.now(),
            files_processed=100,
            files_pending=50,
            files_failed=15,  # 15% error rate
            current_throughput=5.0,  # Low throughput
            average_processing_time=2.0,
            memory_usage=90.0,  # High memory
            cpu_usage=95.0,  # High CPU
            active_workers=4,
            queue_depth=1500,  # High queue
            cache_hits=10,
            cache_misses=90,
            api_calls=100,
            api_errors=5,
        )

        dashboard._check_alerts(metrics)

        # Should have multiple alerts
        self.assertGreater(len(dashboard.active_alerts), 0)

        # Check alert types
        alert_types = [alert["type"] for alert in dashboard.active_alerts]
        self.assertIn("error_rate", alert_types)
        self.assertIn("memory_usage", alert_types)
        self.assertIn("cpu_usage", alert_types)
        self.assertIn("low_throughput", alert_types)
        self.assertIn("queue_depth", alert_types)

    def test_trend_calculation(self):
        """Test trend calculation."""
        dashboard = MonitoringDashboard()

        # Add historical metrics
        for i in range(20):
            metrics = LiveMetrics(
                timestamp=datetime.now(),
                files_processed=i * 10,
                files_pending=100 - i * 5,
                files_failed=i,
                current_throughput=10.0 + i,  # Increasing
                average_processing_time=2.0,
                memory_usage=50.0 + i * 2,  # Increasing
                cpu_usage=40.0,
                active_workers=4,
                queue_depth=100,
                cache_hits=i * 2,
                cache_misses=i,
                api_calls=i * 5,
                api_errors=i % 3,
            )
            dashboard.metrics_history.append(metrics)

        # Calculate trends
        throughput_trend = dashboard._calculate_trend("current_throughput")
        memory_trend = dashboard._calculate_trend("memory_usage")

        self.assertEqual(throughput_trend, "increasing")
        self.assertEqual(memory_trend, "increasing")

    def test_performance_summary(self):
        """Test performance summary generation."""
        dashboard = MonitoringDashboard()

        # Add a metric
        metrics = LiveMetrics(
            timestamp=datetime.now(),
            files_processed=50,
            files_pending=20,
            files_failed=2,
            current_throughput=30.0,
            average_processing_time=2.0,
            memory_usage=65.0,
            cpu_usage=45.0,
            active_workers=4,
            queue_depth=20,
            cache_hits=40,
            cache_misses=10,
            api_calls=50,
            api_errors=2,
        )
        dashboard.metrics_history.append(metrics)

        summary = dashboard.get_performance_summary()

        self.assertIn("current_state", summary)
        self.assertIn("performance", summary)
        self.assertIn("cache_performance", summary)
        self.assertIn("api_performance", summary)
        self.assertIn("estimates", summary)

        # Check cache hit rate calculation
        self.assertAlmostEqual(
            summary["cache_performance"]["hit_rate"],
            0.8,  # 40 hits / 50 total
            places=2,
        )

    def test_html_generation(self):
        """Test HTML dashboard generation."""
        dashboard = MonitoringDashboard()

        # Add test data
        metrics = LiveMetrics(
            timestamp=datetime.now(),
            files_processed=100,
            files_pending=50,
            files_failed=5,
            current_throughput=25.0,
            average_processing_time=2.4,
            memory_usage=70.0,
            cpu_usage=55.0,
            active_workers=8,
            queue_depth=50,
            cache_hits=80,
            cache_misses=20,
            api_calls=100,
            api_errors=3,
        )
        dashboard.metrics_history.append(metrics)

        # Generate HTML
        html = dashboard.generate_dashboard_html()

        # Check content
        self.assertIn("Pipeline Performance Dashboard", html)
        self.assertIn("Files Processed", html)
        self.assertIn("100", html)  # Files processed count
        self.assertIn("Throughput", html)
        self.assertIn("25.0", html)  # Throughput value


if __name__ == "__main__":
    unittest.main()
