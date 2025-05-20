"""
Performance optimization module for high-volume document processing.
Optimizes the OCR + AI pipeline for processing 400+ Hansman Syracuse documents.
"""

import logging
import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import threading
from datetime import datetime
import multiprocessing as mp
import numpy as np

from src.metadata_extraction.ocr_ai_pipeline import OCRAIPipeline, PipelineResult
from src.file_access.ocr_processor import OCRProcessor
from src.metadata_extraction.ai_summarizer import AISummarizer
from src.claude_integration.client import ClaudeClient

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline optimization."""

    total_files: int
    processed_files: int
    failed_files: int
    total_time: float
    average_time_per_file: float
    peak_memory_usage: float  # MB
    cpu_utilization: float  # Percentage
    throughput: float  # Files per second
    ocr_time: float
    ai_time: float
    io_time: float
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime


class ResourceMonitor:
    """Monitor system resources during processing."""

    def __init__(self):
        self.metrics = {"memory": [], "cpu": [], "disk_io": []}
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            self.metrics["memory"].append(psutil.virtual_memory().percent)
            self.metrics["cpu"].append(psutil.cpu_percent(interval=0.1))

            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                self.metrics["disk_io"].append(
                    {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                    }
                )
            except Exception:
                pass

            time.sleep(1)  # Sample every second

    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage."""
        return {
            "peak_memory": max(self.metrics["memory"]) if self.metrics["memory"] else 0,
            "peak_cpu": max(self.metrics["cpu"]) if self.metrics["cpu"] else 0,
            "average_cpu": np.mean(self.metrics["cpu"]) if self.metrics["cpu"] else 0,
            "average_memory": (
                np.mean(self.metrics["memory"]) if self.metrics["memory"] else 0
            ),
        }


class DocumentBatcher:
    """Batch documents for optimal processing."""

    def __init__(self, batch_size: int = 10, max_batch_memory: int = 500):  # MB
        self.batch_size = batch_size
        self.max_batch_memory = max_batch_memory

    def create_batches(
        self, file_paths: List[Path], prioritize_by_size: bool = True
    ) -> List[List[Path]]:
        """Create optimized batches of documents."""
        if prioritize_by_size:
            # Sort by file size (smaller files first for better parallelization)
            sorted_files = sorted(file_paths, key=lambda p: p.stat().st_size)
        else:
            sorted_files = file_paths

        batches = []
        current_batch = []
        current_size = 0

        for file_path in sorted_files:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            # Check if adding this file would exceed batch limits
            if (
                len(current_batch) >= self.batch_size
                or current_size + file_size_mb > self.max_batch_memory
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [file_path]
                current_size = file_size_mb
            else:
                current_batch.append(file_path)
                current_size += file_size_mb

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches


class OptimizedPipeline:
    """Optimized pipeline for high-volume document processing."""

    def __init__(
        self,
        ocr_processor: Optional[OCRProcessor] = None,
        ai_summarizer: Optional[AISummarizer] = None,
        claude_client: Optional[ClaudeClient] = None,
        max_workers: int = None,
        use_multiprocessing: bool = True,
        cache_enabled: bool = True,
        batch_size: int = 10,
        api_rate_limit: int = 50,  # Requests per minute
    ):
        """Initialize optimized pipeline.

        Args:
            ocr_processor: OCR processor instance
            ai_summarizer: AI summarizer instance
            claude_client: Claude client
            max_workers: Maximum parallel workers (auto-detect if None)
            use_multiprocessing: Use process pool for OCR
            cache_enabled: Enable result caching
            batch_size: Batch size for processing
            api_rate_limit: API rate limit per minute
        """
        self.base_pipeline = OCRAIPipeline(
            ocr_processor=ocr_processor,
            ai_summarizer=ai_summarizer,
            claude_client=claude_client,
            cache_results=cache_enabled,
        )

        # Determine optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            self.max_workers = min(cpu_count * 2, 16)  # 2x CPUs, max 16
        else:
            self.max_workers = max_workers

        self.use_multiprocessing = use_multiprocessing
        self.batch_size = batch_size
        self.api_rate_limit = api_rate_limit

        # Performance tracking
        self.resource_monitor = ResourceMonitor()
        self.metrics = {
            "ocr_times": [],
            "ai_times": [],
            "total_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": [],
        }

        # Rate limiting for API calls
        self.api_semaphore = threading.Semaphore(api_rate_limit)
        self.api_call_times = []

        logger.info(f"Optimized pipeline initialized with {self.max_workers} workers")

    def process_documents(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable] = None,
        optimization_level: str = "balanced",
    ) -> Tuple[List[PipelineResult], PerformanceMetrics]:
        """Process documents with optimization.

        Args:
            file_paths: List of files to process
            progress_callback: Progress callback function
            optimization_level: "speed", "balanced", or "quality"

        Returns:
            Tuple of (results, performance metrics)
        """
        start_time = time.time()
        self.resource_monitor.start_monitoring()

        try:
            # Configure optimization
            config = self._get_optimization_config(optimization_level)

            # Create batches
            batcher = DocumentBatcher(
                batch_size=config["batch_size"],
                max_batch_memory=config["max_batch_memory"],
            )
            batches = batcher.create_batches(file_paths)

            logger.info(f"Processing {len(file_paths)} files in {len(batches)} batches")

            # Process batches
            results = []
            completed = 0

            for batch_idx, batch in enumerate(batches):
                batch_results = self._process_batch(
                    batch, config, batch_idx, len(batches)
                )

                results.extend(batch_results)
                completed += len(batch_results)

                if progress_callback:
                    progress_callback(completed, len(file_paths))

            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time

            metrics = self._calculate_metrics(results, total_time, len(file_paths))

            return results, metrics

        finally:
            self.resource_monitor.stop_monitoring()

    def _get_optimization_config(self, level: str) -> Dict[str, Any]:
        """Get optimization configuration based on level."""
        configs = {
            "speed": {
                "batch_size": 20,
                "max_batch_memory": 1000,
                "ocr_timeout": 30,
                "ai_timeout": 30,
                "retry_attempts": 1,
                "use_multiprocessing": True,
                "parallel_ocr": True,
                "parallel_ai": True,
                "cache_aggressive": True,
            },
            "balanced": {
                "batch_size": 10,
                "max_batch_memory": 500,
                "ocr_timeout": 60,
                "ai_timeout": 60,
                "retry_attempts": 2,
                "use_multiprocessing": True,
                "parallel_ocr": True,
                "parallel_ai": False,  # Serialize AI calls for rate limiting
                "cache_aggressive": True,
            },
            "quality": {
                "batch_size": 5,
                "max_batch_memory": 250,
                "ocr_timeout": 120,
                "ai_timeout": 120,
                "retry_attempts": 3,
                "use_multiprocessing": False,  # Single process for consistency
                "parallel_ocr": False,
                "parallel_ai": False,
                "cache_aggressive": False,
            },
        }

        return configs.get(level, configs["balanced"])

    def _process_batch(
        self,
        batch: List[Path],
        config: Dict[str, Any],
        batch_idx: int,
        total_batches: int,
    ) -> List[PipelineResult]:
        """Process a batch of documents."""
        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} files)"
        )

        if config["parallel_ocr"] and config["use_multiprocessing"]:
            # Process OCR in parallel using multiprocessing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                ocr_futures = {
                    executor.submit(self._process_ocr, file_path): file_path
                    for file_path in batch
                }

                ocr_results = {}
                for future in as_completed(ocr_futures):
                    file_path = ocr_futures[future]
                    try:
                        ocr_result = future.result()
                        ocr_results[file_path] = ocr_result
                    except Exception as e:
                        logger.error(f"OCR error for {file_path}: {e}")
                        ocr_results[file_path] = None
        else:
            # Process OCR sequentially
            ocr_results = {}
            for file_path in batch:
                try:
                    ocr_result = self._process_ocr(file_path)
                    ocr_results[file_path] = ocr_result
                except Exception as e:
                    logger.error(f"OCR error for {file_path}: {e}")
                    ocr_results[file_path] = None

        # Process AI summarization (rate-limited)
        results = []

        if config["parallel_ai"]:
            # Parallel AI processing with rate limiting
            with ThreadPoolExecutor(max_workers=min(self.max_workers, 4)) as executor:
                ai_futures = []

                for file_path, ocr_result in ocr_results.items():
                    if ocr_result:
                        future = executor.submit(
                            self._process_ai_with_rate_limit,
                            file_path,
                            ocr_result,
                            config,
                        )
                        ai_futures.append((future, file_path))

                for future, file_path in ai_futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"AI error for {file_path}: {e}")
                        results.append(self._create_error_result(file_path, str(e)))
        else:
            # Sequential AI processing
            for file_path, ocr_result in ocr_results.items():
                if ocr_result:
                    try:
                        result = self._process_ai_with_rate_limit(
                            file_path, ocr_result, config
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"AI error for {file_path}: {e}")
                        results.append(self._create_error_result(file_path, str(e)))

        return results

    def _process_ocr(self, file_path: Path) -> Optional[Any]:
        """Process OCR for a single file."""
        start_time = time.time()

        try:
            result = self.base_pipeline.ocr_processor.process_file(file_path)

            # Track metrics
            ocr_time = time.time() - start_time
            self.metrics["ocr_times"].append(ocr_time)

            return result

        except Exception as e:
            logger.error(f"OCR processing error for {file_path}: {e}")
            self.metrics["errors"].append(str(e))
            return None

    def _process_ai_with_rate_limit(
        self, file_path: Path, ocr_result: Any, config: Dict[str, Any]
    ) -> PipelineResult:
        """Process AI summarization with rate limiting."""
        # Check rate limit
        self._enforce_rate_limit()

        start_time = time.time()

        try:
            # Check cache first
            file_hash = self.base_pipeline._calculate_file_hash(file_path)
            cached_result = self.base_pipeline._check_cache(file_hash)

            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result

            self.metrics["cache_misses"] += 1

            # Process AI summarization
            ai_summary = self.base_pipeline.ai_summarizer.summarize_document(
                file_path=file_path,
                ocr_text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
            )

            # Create pipeline result
            result = PipelineResult(
                file_path=file_path,
                ocr_result=ocr_result,
                ai_summary=ai_summary,
                processing_time=time.time() - start_time,
                success=True,
                file_hash=file_hash,
            )

            # Cache result
            if config.get("cache_aggressive", True):
                self.base_pipeline._cache_result(file_hash, result)

            # Track metrics
            ai_time = time.time() - start_time
            self.metrics["ai_times"].append(ai_time)

            return result

        except Exception as e:
            logger.error(f"AI processing error for {file_path}: {e}")
            self.metrics["errors"].append(str(e))
            return self._create_error_result(file_path, str(e))

    def _enforce_rate_limit(self):
        """Enforce API rate limiting."""
        current_time = time.time()

        # Remove old API calls (older than 1 minute)
        self.api_call_times = [t for t in self.api_call_times if current_time - t < 60]

        # Check if we need to wait
        if len(self.api_call_times) >= self.api_rate_limit:
            wait_time = 60 - (current_time - self.api_call_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        # Record this API call
        self.api_call_times.append(current_time)

    def _create_error_result(self, file_path: Path, error_msg: str) -> PipelineResult:
        """Create error result for failed processing."""
        from src.file_access.ocr_processor import OCRResult
        from src.metadata_extraction.ai_summarizer import DocumentSummary

        return PipelineResult(
            file_path=file_path,
            ocr_result=OCRResult(
                text="",
                confidence=0.0,
                engine_used="none",
                processing_time=0.0,
                error_message=error_msg,
            ),
            ai_summary=DocumentSummary(
                file_path=str(file_path),
                ocr_text="",
                summary="Processing failed",
                category="error",
                confidence_score=0.0,
                key_entities={},
                date_references=[],
                photo_subjects=[],
                location_references=[],
                content_type="unknown",
                historical_period="unknown",
                classification_tags=[],
                claude_metadata={},
                processing_time=0.0,
                error_message=error_msg,
            ),
            processing_time=0.0,
            success=False,
            error_message=error_msg,
        )

    def _calculate_metrics(
        self, results: List[PipelineResult], total_time: float, total_files: int
    ) -> PerformanceMetrics:
        """Calculate performance metrics."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        # Get resource metrics
        resource_metrics = self.resource_monitor.get_peak_metrics()

        # Calculate timing metrics
        avg_ocr_time = (
            np.mean(self.metrics["ocr_times"]) if self.metrics["ocr_times"] else 0
        )
        avg_ai_time = (
            np.mean(self.metrics["ai_times"]) if self.metrics["ai_times"] else 0
        )
        avg_total_time = total_time / total_files if total_files > 0 else 0

        # Calculate rates
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total if cache_total > 0 else 0
        )
        error_rate = len(failed_results) / len(results) if results else 0
        throughput = len(results) / total_time if total_time > 0 else 0

        return PerformanceMetrics(
            total_files=total_files,
            processed_files=len(successful_results),
            failed_files=len(failed_results),
            total_time=total_time,
            average_time_per_file=avg_total_time,
            peak_memory_usage=resource_metrics["peak_memory"],
            cpu_utilization=resource_metrics["average_cpu"],
            throughput=throughput,
            ocr_time=avg_ocr_time,
            ai_time=avg_ai_time,
            io_time=avg_total_time - avg_ocr_time - avg_ai_time,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate,
            timestamp=datetime.now(),
        )

    def analyze_bottlenecks(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        analysis = {"bottlenecks": [], "recommendations": [], "performance_grade": "A"}

        # Analyze timing
        total_processing = metrics.ocr_time + metrics.ai_time
        ocr_percentage = (
            (metrics.ocr_time / total_processing * 100) if total_processing > 0 else 0
        )
        ai_percentage = (
            (metrics.ai_time / total_processing * 100) if total_processing > 0 else 0
        )

        if ocr_percentage > 70:
            analysis["bottlenecks"].append("OCR processing")
            analysis["recommendations"].append(
                "Consider using more OCR workers or GPU acceleration"
            )

        if ai_percentage > 70:
            analysis["bottlenecks"].append("AI summarization")
            analysis["recommendations"].append(
                "Increase API rate limits or use batch API calls"
            )

        # Analyze resource usage
        if metrics.peak_memory_usage > 80:
            analysis["bottlenecks"].append("Memory usage")
            analysis["recommendations"].append(
                "Reduce batch sizes or optimize memory usage"
            )

        if metrics.cpu_utilization < 50:
            analysis["recommendations"].append(
                "Increase parallelization to better utilize CPU"
            )
        elif metrics.cpu_utilization > 90:
            analysis["bottlenecks"].append("CPU saturation")
            analysis["recommendations"].append("Consider using more powerful hardware")

        # Analyze cache performance
        if metrics.cache_hit_rate < 0.2:
            analysis["recommendations"].append(
                "Enable or optimize caching for better performance"
            )

        # Analyze error rate
        if metrics.error_rate > 0.1:
            analysis["bottlenecks"].append("High error rate")
            analysis["recommendations"].append("Investigate and fix processing errors")

        # Determine performance grade
        if metrics.throughput < 0.5:  # Less than 30 files per minute
            analysis["performance_grade"] = "D"
        elif metrics.throughput < 1.0:  # Less than 60 files per minute
            analysis["performance_grade"] = "C"
        elif metrics.throughput < 2.0:  # Less than 120 files per minute
            analysis["performance_grade"] = "B"
        else:
            analysis["performance_grade"] = "A"

        return analysis

    def generate_performance_report(
        self, metrics: PerformanceMetrics, output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks(metrics)

        report = {
            "summary": {
                "total_files": metrics.total_files,
                "processed_successfully": metrics.processed_files,
                "failed": metrics.failed_files,
                "total_time_seconds": metrics.total_time,
                "throughput_files_per_minute": metrics.throughput * 60,
                "performance_grade": bottleneck_analysis["performance_grade"],
            },
            "timing_breakdown": {
                "average_total_time": metrics.average_time_per_file,
                "average_ocr_time": metrics.ocr_time,
                "average_ai_time": metrics.ai_time,
                "average_io_time": metrics.io_time,
                "ocr_percentage": (
                    (metrics.ocr_time / metrics.average_time_per_file * 100)
                    if metrics.average_time_per_file > 0
                    else 0
                ),
                "ai_percentage": (
                    (metrics.ai_time / metrics.average_time_per_file * 100)
                    if metrics.average_time_per_file > 0
                    else 0
                ),
            },
            "resource_usage": {
                "peak_memory_percent": metrics.peak_memory_usage,
                "average_cpu_percent": metrics.cpu_utilization,
                "cache_hit_rate": metrics.cache_hit_rate,
            },
            "error_analysis": {
                "error_rate": metrics.error_rate,
                "total_errors": len(self.metrics.get("errors", [])),
                "error_types": self._categorize_errors(),
            },
            "bottleneck_analysis": bottleneck_analysis,
            "optimization_recommendations": self._generate_recommendations(
                metrics, bottleneck_analysis
            ),
            "timestamp": metrics.timestamp.isoformat(),
        }

        # Save report if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {output_path}")

        return report

    def _categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type."""
        error_categories = {}

        for error in self.metrics.get("errors", []):
            error_type = "unknown"

            if "OCR" in error:
                error_type = "ocr_error"
            elif "API" in error or "rate limit" in error.lower():
                error_type = "api_error"
            elif "timeout" in error.lower():
                error_type = "timeout_error"
            elif "memory" in error.lower():
                error_type = "memory_error"

            error_categories[error_type] = error_categories.get(error_type, 0) + 1

        return error_categories

    def _generate_recommendations(
        self, metrics: PerformanceMetrics, bottleneck_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = bottleneck_analysis["recommendations"].copy()

        # Specific recommendations based on metrics
        if metrics.throughput < 1.0:  # Less than 60 files per minute
            recommendations.append(
                "For 400+ files, current throughput will take over 7 hours"
            )
            recommendations.append(
                "Consider processing in multiple sessions or overnight"
            )

        if metrics.ocr_time > metrics.ai_time * 2:
            recommendations.append(
                "OCR is the primary bottleneck - consider GPU-accelerated OCR"
            )

        if metrics.cache_hit_rate < 0.1:
            recommendations.append("Very low cache usage - ensure caching is enabled")

        # Resource-based recommendations
        if metrics.peak_memory_usage > 70:
            recommendations.append(
                f"High memory usage ({metrics.peak_memory_usage:.1f}%) - monitor for OOM errors"
            )

        # Scale recommendations for 400+ files
        estimated_time_hours = (
            400 / (metrics.throughput * 3600)
            if metrics.throughput > 0
            else float("inf")
        )
        if estimated_time_hours > 4:
            recommendations.append(
                f"Estimated time for 400 files: {estimated_time_hours:.1f} hours"
            )
            recommendations.append(
                "Consider cloud-based processing for better scalability"
            )

        return recommendations
