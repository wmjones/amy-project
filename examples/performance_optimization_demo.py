#!/usr/bin/env python3
"""
Demo script for performance optimization and scalability testing.
Shows how to process large batches of documents efficiently.
"""

import sys
import os
from pathlib import Path
import json
import logging
import time
import threading
from typing import List
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.performance_optimizer import (
    OptimizedPipeline,
    PerformanceMetrics,
    ResourceMonitor,
)
from src.optimization.monitoring_dashboard import MonitoringDashboard, ConsoleDashboard
from src.file_access.ocr_processor import OCRProcessor
from src.metadata_extraction.ai_summarizer import AISummarizer
from src.claude_integration.client import ClaudeClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_documents(num_documents: int = 50) -> List[Path]:
    """Create test documents for performance testing."""
    test_dir = Path("./performance_test_data")
    test_dir.mkdir(exist_ok=True)

    test_files = []

    for i in range(num_documents):
        # Create different types of test documents
        if i % 3 == 0:
            filename = f"certificate_{i:04d}.txt"
            content = f"""
            Syracuse Salt Company
            Certificate of Stock No. {i}

            This certifies that Holder {i} is the owner of {50 + i} shares
            of the capital stock of the Syracuse Salt Company.

            Dated this {(i % 28) + 1} day of March, {1890 + (i % 20)}
            """
        elif i % 3 == 1:
            filename = f"photo_{i:04d}.txt"
            content = f"""
            [Photograph {i}]
            Black and white photograph showing location {i % 5}
            in Syracuse, New York.

            Date: {1900 + (i % 50)}
            Location: {['Erie Canal', 'Clinton Square', 'Armory Square', 'University Hill', 'Eastwood'][i % 5]}
            """
        else:
            filename = f"letter_{i:04d}.txt"
            content = f"""
            Dear Friend {i},

            I hope this letter finds you well. The weather here in Syracuse
            has been quite pleasant this {['spring', 'summer', 'fall', 'winter'][i % 4]}.

            Sincerely,
            Writer {i}

            Date: {['January', 'April', 'July', 'October'][i % 4]} {(i % 28) + 1}, {1950 + (i % 30)}
            """

        file_path = test_dir / filename
        file_path.write_text(content)
        test_files.append(file_path)

    return test_files


def demo_performance_optimization():
    """Demonstrate performance optimization for high-volume processing."""
    print("=" * 60)
    print("Performance Optimization Demo")
    print("=" * 60)

    # Check for API key
    use_real_api = os.getenv("ANTHROPIC_API_KEY") is not None
    if not use_real_api:
        print("\nNote: No Anthropic API key found. Using mock processing.")

    # Initialize components
    print("\n1. Initializing optimized pipeline...")

    try:
        # Initialize base components
        ocr_processor = OCRProcessor()
        ai_summarizer = None
        claude_client = None

        if use_real_api:
            claude_client = ClaudeClient()
            ai_summarizer = AISummarizer(claude_client=claude_client)

        # Initialize optimized pipeline
        pipeline = OptimizedPipeline(
            ocr_processor=ocr_processor,
            ai_summarizer=ai_summarizer,
            claude_client=claude_client,
            max_workers=8,
            use_multiprocessing=True,
            cache_enabled=True,
            batch_size=10,
            api_rate_limit=30,  # Lower for demo
        )

        print("✓ Optimized pipeline initialized")

    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        return

    # Create test documents
    print("\n2. Creating test documents...")
    test_files = create_test_documents(50)
    print(f"✓ Created {len(test_files)} test documents")

    # Initialize monitoring dashboard
    print("\n3. Starting monitoring dashboard...")
    dashboard = MonitoringDashboard(update_interval=2)

    # Add console display callback
    dashboard.add_update_callback(ConsoleDashboard.display_metrics)

    # Start dashboard
    dashboard.start()

    # Demonstrate different optimization levels
    optimization_levels = ["speed", "balanced", "quality"]

    for level in optimization_levels:
        print(f"\n{'-' * 50}")
        print(f"4. Testing with optimization level: {level}")
        print(f"{'-' * 50}")

        # Progress callback
        def progress_callback(completed, total):
            # Update dashboard metrics
            dashboard.update_queue_metrics(
                pending=total - completed,
                active_workers=pipeline.max_workers,
                queue_depth=total - completed,
            )

        # Monitor system resources
        def monitor_resources():
            while processing:
                try:
                    dashboard.update_system_metrics(
                        {
                            "memory": psutil.virtual_memory().percent,
                            "cpu": psutil.cpu_percent(interval=0.5),
                        }
                    )
                    time.sleep(1)
                except:
                    pass

        processing = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Process documents
        start_time = time.time()

        if use_real_api or level == "speed":  # Only test speed mode without API
            results, metrics = pipeline.process_documents(
                test_files[:20],  # Limit to 20 for demo
                progress_callback=progress_callback,
                optimization_level=level,
            )
        else:
            # Mock results for demo
            results = []
            for i, file_path in enumerate(test_files[:20]):
                time.sleep(0.1)  # Simulate processing

                # Create mock result
                from src.metadata_extraction.ocr_ai_pipeline import PipelineResult
                from src.file_access.ocr_processor import OCRResult
                from src.metadata_extraction.ai_summarizer import DocumentSummary

                mock_result = PipelineResult(
                    file_path=file_path,
                    ocr_result=OCRResult(
                        text=file_path.read_text(),
                        confidence=0.9,
                        engine_used="mock",
                        processing_time=0.1,
                    ),
                    ai_summary=DocumentSummary(
                        file_path=str(file_path),
                        ocr_text=file_path.read_text(),
                        summary="Mock summary",
                        category="document",
                        confidence_score=0.85,
                        key_entities={},
                        date_references=[],
                        photo_subjects=[],
                        location_references=["Syracuse"],
                        content_type="document",
                        historical_period="modern",
                        classification_tags=["test"],
                        claude_metadata={},
                        processing_time=0.5,
                    ),
                    processing_time=0.6,
                    success=True,
                )
                results.append(mock_result)

                # Update progress
                progress_callback(i + 1, 20)

                # Record metrics
                dashboard.record_event(
                    event_type="process",
                    processing_time=0.6,
                    success=True,
                    cache_hit=i % 3 == 0,
                    api_called=True,
                )

            # Create mock metrics
            metrics = PerformanceMetrics(
                total_files=20,
                processed_files=20,
                failed_files=0,
                total_time=time.time() - start_time,
                average_time_per_file=0.6,
                peak_memory_usage=65.5,
                cpu_utilization=45.2,
                throughput=20 / (time.time() - start_time),
                ocr_time=0.3,
                ai_time=0.2,
                io_time=0.1,
                cache_hit_rate=0.33,
                error_rate=0.0,
                timestamp=datetime.now(),
            )

        processing = False

        # Display metrics
        print(f"\nResults for {level} optimization:")
        print(f"Total files: {metrics.total_files}")
        print(f"Successful: {metrics.processed_files}")
        print(f"Failed: {metrics.failed_files}")
        print(f"Total time: {metrics.total_time:.1f}s")
        print(f"Throughput: {metrics.throughput:.2f} files/second")
        print(f"Average time per file: {metrics.average_time_per_file:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.1f}%")
        print(f"CPU usage: {metrics.cpu_utilization:.1f}%")

        # Analyze bottlenecks
        analysis = pipeline.analyze_bottlenecks(metrics)

        print(f"\nPerformance grade: {analysis['performance_grade']}")

        if analysis["bottlenecks"]:
            print(f"Bottlenecks: {', '.join(analysis['bottlenecks'])}")

        if analysis["recommendations"]:
            print("\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"  - {rec}")

        # Wait a bit before next test
        time.sleep(3)

    # Test with 400+ documents simulation
    print("\n" + "=" * 60)
    print("5. Simulating 400+ document processing")
    print("=" * 60)

    # Create performance report
    report = pipeline.generate_performance_report(metrics)

    print("\nProjected performance for 400 files:")
    estimated_time = 400 / metrics.throughput if metrics.throughput > 0 else 0
    print(
        f"Estimated time: {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)"
    )
    print(f"Estimated completion: {estimated_time/3600:.1f} hours")

    print("\nOptimization recommendations for 400+ files:")
    for rec in report["optimization_recommendations"]:
        print(f"  - {rec}")

    # Generate and save reports
    print("\n6. Generating reports...")

    # Save performance report
    report_path = Path("./performance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Performance report saved to: {report_path}")

    # Get dashboard summary
    summary = dashboard.get_performance_summary()
    summary_path = Path("./dashboard_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Dashboard summary saved to: {summary_path}")

    # Generate HTML dashboard
    html_path = Path("./performance_dashboard.html")
    dashboard.generate_dashboard_html(html_path)
    print(f"✓ HTML dashboard saved to: {html_path}")

    # Export metrics history
    metrics_path = Path("./metrics_history.json")
    dashboard.export_metrics(metrics_path)
    print(f"✓ Metrics history exported to: {metrics_path}")

    # Stop dashboard
    dashboard.stop()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def test_resource_monitoring():
    """Test resource monitoring functionality."""
    print("\n" + "=" * 50)
    print("Testing Resource Monitoring")
    print("=" * 50)

    monitor = ResourceMonitor()
    monitor.start_monitoring()

    print("\nMonitoring system resources for 10 seconds...")

    # Simulate some load
    for i in range(10):
        _ = [j**2 for j in range(1000000)]  # CPU load
        time.sleep(1)
        print(
            f"  {i+1}s - Memory: {psutil.virtual_memory().percent:.1f}%, "
            f"CPU: {psutil.cpu_percent(interval=0.1):.1f}%"
        )

    monitor.stop_monitoring()

    # Get peak metrics
    peak_metrics = monitor.get_peak_metrics()

    print("\nPeak metrics:")
    print(f"  Peak memory: {peak_metrics['peak_memory']:.1f}%")
    print(f"  Peak CPU: {peak_metrics['peak_cpu']:.1f}%")
    print(f"  Average memory: {peak_metrics['average_memory']:.1f}%")
    print(f"  Average CPU: {peak_metrics['average_cpu']:.1f}%")


def cleanup_test_data():
    """Clean up test data."""
    import shutil

    test_dir = Path("./performance_test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\nTest data cleaned up.")


if __name__ == "__main__":
    try:
        # Run main demo
        demo_performance_optimization()

        # Test resource monitoring
        print("\n" + "=" * 60)
        test_resource_monitoring()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        logger.exception("Demo failed with error")
    finally:
        # Clean up
        response = input("\nClean up test data? (y/n): ")
        if response.lower() == "y":
            cleanup_test_data()
