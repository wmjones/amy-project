"""
Integration pipeline for OCR and AI-driven summarization.
Connects OCR processing with Claude AI for comprehensive document analysis.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from src.file_access.ocr_processor import OCRProcessor, OCRResult, OCRCache
from src.metadata_extraction.ai_summarizer import AISummarizer, DocumentSummary
from src.claude_integration.client import ClaudeClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from OCR + AI pipeline processing."""

    file_path: Path
    ocr_result: OCRResult
    ai_summary: DocumentSummary
    processing_time: float
    cache_hit: bool = False
    success: bool = True
    error_message: Optional[str] = None
    file_hash: str = ""
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "ocr_result": {
                "text": self.ocr_result.text,
                "confidence": self.ocr_result.confidence,
                "engine_used": self.ocr_result.engine_used,
                "page_count": self.ocr_result.page_count,
                "cached": self.ocr_result.cached,
            },
            "ai_summary": {
                "summary": self.ai_summary.summary,
                "category": self.ai_summary.category,
                "confidence_score": self.ai_summary.confidence_score,
                "key_entities": self.ai_summary.key_entities,
                "historical_period": self.ai_summary.historical_period,
                "suggested_folder_path": self.ai_summary.suggested_folder_path,
            },
            "processing_time": self.processing_time,
            "cache_hit": self.cache_hit,
            "success": self.success,
            "error_message": self.error_message,
            "file_hash": self.file_hash,
        }


class OCRAIPipeline:
    """Integrated pipeline for OCR and AI processing of historical documents."""

    def __init__(
        self,
        ocr_processor: Optional[OCRProcessor] = None,
        ai_summarizer: Optional[AISummarizer] = None,
        claude_client: Optional[ClaudeClient] = None,
        cache_results: bool = True,
        confidence_threshold: float = 0.6,
    ):
        """Initialize the integrated pipeline.

        Args:
            ocr_processor: OCR processor instance
            ai_summarizer: AI summarizer instance
            claude_client: Claude client for AI operations
            cache_results: Whether to cache results
            confidence_threshold: Minimum confidence for AI processing
        """
        self.ocr_processor = ocr_processor or OCRProcessor()
        self.ai_summarizer = ai_summarizer or AISummarizer(claude_client=claude_client)
        self.cache_results = cache_results
        self.confidence_threshold = confidence_threshold

        # Initialize result cache
        self.result_cache = {}
        self.cache_dir = Path("./pipeline_cache")
        self.cache_dir.mkdir(exist_ok=True)

        logger.info("Initialized OCR + AI pipeline")

    def process_file(
        self,
        file_path: Path,
        file_metadata: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False,
    ) -> PipelineResult:
        """Process a single file through the complete pipeline.

        Args:
            file_path: Path to the file
            file_metadata: Optional metadata about the file
            force_reprocess: Skip cache and force reprocessing

        Returns:
            PipelineResult with complete analysis
        """
        start_time = time.time()

        try:
            # Calculate file hash for caching
            file_hash = self._calculate_file_hash(file_path)

            # Check cache if not forcing reprocess
            if not force_reprocess:
                cached_result = self._check_cache(file_hash)
                if cached_result:
                    logger.info(f"Cache hit for {file_path}")
                    return cached_result

            # Step 1: OCR Processing
            logger.info(f"Processing OCR for {file_path}")
            ocr_result = self.ocr_processor.process_file(file_path, file_metadata)

            # Check if OCR was successful and has sufficient confidence
            if not ocr_result.text or ocr_result.confidence < self.confidence_threshold:
                logger.warning(
                    f"Low OCR confidence ({ocr_result.confidence}) for {file_path}"
                )

            # Step 2: AI Summarization (proceed even with low confidence)
            logger.info(f"Running AI summarization for {file_path}")
            ai_summary = self.ai_summarizer.summarize_document(
                file_path=file_path,
                ocr_text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
                additional_context=file_metadata,
            )

            # Create pipeline result
            processing_time = time.time() - start_time
            result = PipelineResult(
                file_path=file_path,
                ocr_result=ocr_result,
                ai_summary=ai_summary,
                processing_time=processing_time,
                cache_hit=False,
                success=True,
                file_hash=file_hash,
                metadata=file_metadata,
            )

            # Cache the result if enabled
            if self.cache_results:
                self._cache_result(file_hash, result)

            return result

        except Exception as e:
            logger.error(f"Pipeline error processing {file_path}: {e}")
            processing_time = time.time() - start_time

            # Create error result
            return PipelineResult(
                file_path=file_path,
                ocr_result=OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used="none",
                    processing_time=0.0,
                    error_message=str(e),
                ),
                ai_summary=DocumentSummary(
                    file_path=str(file_path),
                    ocr_text="",
                    summary="Error in pipeline processing",
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
                    error_message=str(e),
                ),
                processing_time=processing_time,
                cache_hit=False,
                success=False,
                error_message=str(e),
                file_hash=file_hash,
            )

    def process_batch(
        self,
        file_paths: List[Path],
        max_workers: int = 4,
        progress_callback: Optional[callable] = None,
    ) -> List[PipelineResult]:
        """Process multiple files through the pipeline.

        Args:
            file_paths: List of file paths to process
            max_workers: Maximum concurrent workers
            progress_callback: Optional progress callback

        Returns:
            List of PipelineResult objects
        """
        results = []
        total_files = len(file_paths)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path
                for file_path in file_paths
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Create error result
                    results.append(self._create_error_result(file_path, str(e)))

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_files)

        return results

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for caching."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _check_cache(self, file_hash: str) -> Optional[PipelineResult]:
        """Check if result is cached."""
        # Check in-memory cache first
        if file_hash in self.result_cache:
            return self.result_cache[file_hash]

        # Check disk cache
        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    # Reconstruct PipelineResult from cached data
                    return self._reconstruct_from_cache(cached_data)
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
                return None

        return None

    def _cache_result(self, file_hash: str, result: PipelineResult):
        """Cache a pipeline result."""
        # In-memory cache
        self.result_cache[file_hash] = result

        # Disk cache
        cache_file = self.cache_dir / f"{file_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")

    def _reconstruct_from_cache(self, cached_data: Dict[str, Any]) -> PipelineResult:
        """Reconstruct PipelineResult from cached data."""
        # This is a simplified reconstruction - in production you'd want
        # to properly reconstruct all nested objects
        return PipelineResult(
            file_path=Path(cached_data["file_path"]),
            ocr_result=OCRResult(
                text=cached_data["ocr_result"]["text"],
                confidence=cached_data["ocr_result"]["confidence"],
                engine_used=cached_data["ocr_result"]["engine_used"],
                processing_time=0.0,  # From cache
                cached=True,
            ),
            ai_summary=DocumentSummary(
                file_path=cached_data["file_path"],
                ocr_text=cached_data["ocr_result"]["text"],
                summary=cached_data["ai_summary"]["summary"],
                category=cached_data["ai_summary"]["category"],
                confidence_score=cached_data["ai_summary"]["confidence_score"],
                key_entities=cached_data["ai_summary"].get("key_entities", {}),
                date_references=[],
                photo_subjects=[],
                location_references=[],
                content_type="unknown",
                historical_period=cached_data["ai_summary"].get(
                    "historical_period", ""
                ),
                classification_tags=[],
                claude_metadata={},
                processing_time=0.0,
            ),
            processing_time=cached_data["processing_time"],
            cache_hit=True,
            success=cached_data["success"],
            error_message=cached_data.get("error_message"),
            file_hash=cached_data["file_hash"],
        )

    def _create_error_result(
        self, file_path: Path, error_message: str
    ) -> PipelineResult:
        """Create an error result for failed processing."""
        return PipelineResult(
            file_path=file_path,
            ocr_result=OCRResult(
                text="",
                confidence=0.0,
                engine_used="none",
                processing_time=0.0,
                error_message=error_message,
            ),
            ai_summary=DocumentSummary(
                file_path=str(file_path),
                ocr_text="",
                summary="Pipeline processing failed",
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
                error_message=error_message,
            ),
            processing_time=0.0,
            cache_hit=False,
            success=False,
            error_message=error_message,
            file_hash="",
        )

    def generate_pipeline_report(
        self, results: List[PipelineResult], output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive report of pipeline processing.

        Args:
            results: List of pipeline results
            output_path: Optional path to save report

        Returns:
            Dictionary containing the pipeline report
        """
        report = {
            "pipeline": "OCR + AI Document Processing",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(results),
            "statistics": {
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "cache_hits": sum(1 for r in results if r.cache_hit),
                "ocr_engines": {},
                "categories": {},
                "confidence_levels": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.5-0.8
                    "low": 0,  # < 0.5
                },
                "total_processing_time": sum(r.processing_time for r in results),
                "average_processing_time": 0.0,
            },
            "errors": [],
            "performance_metrics": {
                "ocr_average_time": 0.0,
                "ai_average_time": 0.0,
                "cache_hit_rate": 0.0,
            },
        }

        # Analyze results
        ocr_times = []
        ai_times = []

        for result in results:
            if not result.success:
                report["errors"].append(
                    {"file": str(result.file_path), "error": result.error_message}
                )
                continue

            # OCR engine usage
            engine = result.ocr_result.engine_used
            report["statistics"]["ocr_engines"][engine] = (
                report["statistics"]["ocr_engines"].get(engine, 0) + 1
            )

            # Document categories
            category = result.ai_summary.category
            report["statistics"]["categories"][category] = (
                report["statistics"]["categories"].get(category, 0) + 1
            )

            # Confidence distribution
            confidence = result.ai_summary.confidence_score
            if confidence > 0.8:
                report["statistics"]["confidence_levels"]["high"] += 1
            elif confidence > 0.5:
                report["statistics"]["confidence_levels"]["medium"] += 1
            else:
                report["statistics"]["confidence_levels"]["low"] += 1

            # Performance metrics
            if result.ocr_result.processing_time:
                ocr_times.append(result.ocr_result.processing_time)
            if result.ai_summary.processing_time:
                ai_times.append(result.ai_summary.processing_time)

        # Calculate averages
        if results:
            report["statistics"]["average_processing_time"] = report["statistics"][
                "total_processing_time"
            ] / len(results)

        if ocr_times:
            report["performance_metrics"]["ocr_average_time"] = sum(ocr_times) / len(
                ocr_times
            )

        if ai_times:
            report["performance_metrics"]["ai_average_time"] = sum(ai_times) / len(
                ai_times
            )

        if results:
            report["performance_metrics"]["cache_hit_rate"] = report["statistics"][
                "cache_hits"
            ] / len(results)

        # Save report if requested
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report

    def export_results(
        self,
        results: List[PipelineResult],
        format: str = "json",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Export processing results in various formats.

        Args:
            results: List of pipeline results
            format: Export format (json, csv, or summary)
            output_dir: Output directory

        Returns:
            Path to the exported file
        """
        if output_dir is None:
            output_dir = Path("./pipeline_exports")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = output_dir / f"pipeline_results_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)

        elif format == "csv":
            import csv

            output_file = output_dir / f"pipeline_results_{timestamp}.csv"

            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(
                    [
                        "File Path",
                        "OCR Confidence",
                        "AI Category",
                        "AI Confidence",
                        "Historical Period",
                        "Suggested Folder",
                        "Processing Time",
                        "Success",
                        "Error",
                    ]
                )

                # Data rows
                for result in results:
                    writer.writerow(
                        [
                            str(result.file_path),
                            result.ocr_result.confidence,
                            result.ai_summary.category,
                            result.ai_summary.confidence_score,
                            result.ai_summary.historical_period,
                            result.ai_summary.suggested_folder_path,
                            result.processing_time,
                            result.success,
                            result.error_message or "",
                        ]
                    )

        elif format == "summary":
            output_file = output_dir / f"pipeline_summary_{timestamp}.txt"

            with open(output_file, "w") as f:
                f.write("OCR + AI Pipeline Processing Summary\n")
                f.write("=" * 40 + "\n\n")

                for result in results:
                    f.write(f"File: {result.file_path.name}\n")
                    f.write(f"Category: {result.ai_summary.category}\n")
                    f.write(f"Summary: {result.ai_summary.summary[:200]}...\n")
                    f.write(f"Confidence: {result.ai_summary.confidence_score:.2f}\n")
                    f.write(
                        f"Suggested Path: {result.ai_summary.suggested_folder_path}\n"
                    )
                    f.write("-" * 40 + "\n\n")

        logger.info(f"Exported results to {output_file}")
        return output_file
