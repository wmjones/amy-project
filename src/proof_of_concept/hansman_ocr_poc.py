#!/usr/bin/env python3
"""
OCR Proof of Concept for Hansman Syracuse Collection.
Implements OCR text extraction with Tesseract and Claude API summarization.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.file_access.ocr_processor import OCRProcessor
from src.file_access.processor import DocumentPreprocessor
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.ai_summarizer import AISummarizer
from dataclasses import dataclass


# Define OCRResult since it's not in the original module
@dataclass
class OCRResult:
    """Container for OCR results."""

    text: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/hansman_poc.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class HansmanOCRProofOfConcept:
    """Proof of concept implementation for Hansman Syracuse Collection OCR."""

    def __init__(self, use_claude: bool = True):
        """Initialize the proof of concept.

        Args:
            use_claude: Whether to use Claude API for summarization
        """
        self.use_claude = use_claude

        # Initialize components
        self.preprocessor = DocumentPreprocessor()
        self.ocr_processor = OCRProcessor(
            language="eng", enhance_contrast=True, denoise=True, deskew=True
        )

        if self.use_claude and os.getenv("ANTHROPIC_API_KEY"):
            self.claude_client = ClaudeClient()
            self.summarizer = AISummarizer(self.claude_client)
        else:
            self.claude_client = None
            self.summarizer = None
            if self.use_claude:
                logger.warning(
                    "Claude API key not found. Summarization will be skipped."
                )

        # Setup directories
        self.setup_directories()

        # Metrics tracking
        self.metrics = {
            "total_files": 0,
            "successful_ocr": 0,
            "failed_ocr": 0,
            "successful_summaries": 0,
            "failed_summaries": 0,
            "processing_times": {},
        }

    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            "data/hansman_samples",
            "output/ocr_results",
            "output/summaries",
            "output/reports",
            "logs",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_sample_files(
        self, sample_dir: str = "data/hansman_samples", limit: int = 10
    ) -> List[Path]:
        """Get sample files from the collection.

        Args:
            sample_dir: Directory containing sample files
            limit: Maximum number of files to process

        Returns:
            List of file paths
        """
        sample_path = Path(sample_dir)

        if not sample_path.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return []

        # Get image files
        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
        files = []

        for ext in image_extensions:
            files.extend(sample_path.glob(f"*{ext}"))
            files.extend(sample_path.glob(f"*{ext.upper()}"))

        # Sort and limit
        files = sorted(files)[:limit]

        logger.info(f"Found {len(files)} sample files to process")
        return files

    def preprocess_image(self, file_path: Path) -> Dict[str, Any]:
        """Preprocess image for OCR.

        Args:
            file_path: Path to image file

        Returns:
            Dictionary with preprocessing results
        """
        start_time = time.time()

        try:
            # Analyze image quality
            metrics = self.preprocessor.analyze_image_quality(str(file_path))

            # Create preprocessing pipeline
            pipeline = self.preprocessor.create_preprocessing_pipeline(metrics)

            # Apply preprocessing
            processed_path = self.preprocessor.preprocess_image(
                str(file_path), output_dir="output/preprocessed"
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "original_path": str(file_path),
                "processed_path": processed_path,
                "quality_metrics": metrics.__dict__,
                "pipeline_steps": pipeline,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Preprocessing error for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def extract_text(
        self, file_path: Path, preprocessed: bool = True
    ) -> Dict[str, Any]:
        """Extract text using OCR.

        Args:
            file_path: Path to image file
            preprocessed: Whether the image has been preprocessed

        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()

        try:
            # Run OCR
            result = self.ocr_processor.process_image(str(file_path))

            processing_time = time.time() - start_time

            return {
                "success": result["success"],
                "text": result["text"],
                "confidence": result.get("confidence", 0),
                "metadata": result.get("metadata", {}),
                "processing_time": processing_time,
                "preprocessed": preprocessed,
            }

        except Exception as e:
            logger.error(f"OCR error for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def summarize_text(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Summarize extracted text using Claude API.

        Args:
            text: Extracted text
            file_path: Original file path for context

        Returns:
            Dictionary with summarization results
        """
        if not self.summarizer or not text.strip():
            return {
                "success": False,
                "error": "No summarizer available or empty text",
                "processing_time": 0,
            }

        start_time = time.time()

        try:
            # Create summary using AI
            summary = self.summarizer.summarize_document(
                file_path=file_path,
                ocr_text=text,
                additional_context={
                    "collection": "Hansman Syracuse",
                    "document_type": "historical_photo",
                },
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "summary": summary.summary,
                "category": summary.category,
                "confidence": summary.confidence_score,
                "key_entities": summary.key_entities,
                "date_references": summary.date_references,
                "location_references": summary.location_references,
                "historical_period": summary.historical_period,
                "suggested_folder": summary.suggested_folder_path,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Summarization error for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file through the entire pipeline.

        Args:
            file_path: Path to file to process

        Returns:
            Complete processing results
        """
        logger.info(f"Processing: {file_path}")

        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

        # Stage 1: Preprocessing
        preprocessing_result = self.preprocess_image(file_path)
        result["stages"]["preprocessing"] = preprocessing_result

        # Stage 2: OCR
        if preprocessing_result["success"]:
            processed_path = Path(preprocessing_result["processed_path"])
            ocr_result = self.extract_text(processed_path, preprocessed=True)
        else:
            ocr_result = self.extract_text(file_path, preprocessed=False)

        result["stages"]["ocr"] = ocr_result

        # Stage 3: Summarization
        if ocr_result["success"] and ocr_result["text"]:
            summary_result = self.summarize_text(ocr_result["text"], file_path)
            result["stages"]["summarization"] = summary_result
        else:
            result["stages"]["summarization"] = {
                "success": False,
                "error": "No text extracted for summarization",
            }

        # Calculate total processing time
        total_time = sum(
            stage.get("processing_time", 0) for stage in result["stages"].values()
        )
        result["total_processing_time"] = total_time

        # Update metrics
        self.update_metrics(result)

        return result

    def update_metrics(self, result: Dict[str, Any]):
        """Update processing metrics.

        Args:
            result: Processing result dictionary
        """
        self.metrics["total_files"] += 1

        if result["stages"]["ocr"]["success"]:
            self.metrics["successful_ocr"] += 1
        else:
            self.metrics["failed_ocr"] += 1

        if result["stages"].get("summarization", {}).get("success"):
            self.metrics["successful_summaries"] += 1
        else:
            self.metrics["failed_summaries"] += 1

        # Track processing times
        file_name = result["file_name"]
        self.metrics["processing_times"][file_name] = {
            "preprocessing": result["stages"]["preprocessing"]["processing_time"],
            "ocr": result["stages"]["ocr"]["processing_time"],
            "summarization": result["stages"]
            .get("summarization", {})
            .get("processing_time", 0),
            "total": result["total_processing_time"],
        }

    def process_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files.

        Args:
            files: List of file paths

        Returns:
            List of processing results
        """
        results = []

        for i, file_path in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")

            result = self.process_single_file(file_path)
            results.append(result)

            # Save individual result
            self.save_result(result)

        return results

    def save_result(self, result: Dict[str, Any]):
        """Save processing result to disk.

        Args:
            result: Processing result dictionary
        """
        # Save OCR result
        if result["stages"]["ocr"]["success"]:
            ocr_file = Path("output/ocr_results") / f"{result['file_name']}.txt"
            ocr_file.write_text(result["stages"]["ocr"]["text"])

        # Save summary
        if result["stages"].get("summarization", {}).get("success"):
            summary_file = (
                Path("output/summaries") / f"{result['file_name']}_summary.json"
            )
            summary_file.write_text(
                json.dumps(result["stages"]["summarization"], indent=2)
            )

        # Save complete result
        result_file = Path("output/reports") / f"{result['file_name']}_complete.json"
        result_file.write_text(json.dumps(result, indent=2))

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive findings report.

        Args:
            results: List of processing results

        Returns:
            Report dictionary
        """
        report = {
            "title": "Hansman Syracuse Collection OCR Proof of Concept Report",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_processed": len(results),
                "successful_ocr": self.metrics["successful_ocr"],
                "failed_ocr": self.metrics["failed_ocr"],
                "ocr_success_rate": self.metrics["successful_ocr"]
                / max(1, self.metrics["total_files"]),
                "successful_summaries": self.metrics["successful_summaries"],
                "failed_summaries": self.metrics["failed_summaries"],
                "summary_success_rate": self.metrics["successful_summaries"]
                / max(1, self.metrics["total_files"]),
            },
            "performance_metrics": self.calculate_performance_metrics(),
            "challenges_and_limitations": self.identify_challenges(results),
            "successful_examples": self.get_successful_examples(results),
            "failed_examples": self.get_failed_examples(results),
            "recommendations": self.generate_recommendations(results),
            "cost_estimates": self.calculate_cost_estimates(),
            "detailed_results": results,
        }

        return report

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from processing times."""
        if not self.metrics["processing_times"]:
            return {}

        # Calculate averages
        all_times = self.metrics["processing_times"].values()

        avg_preprocessing = sum(t["preprocessing"] for t in all_times) / len(all_times)
        avg_ocr = sum(t["ocr"] for t in all_times) / len(all_times)
        avg_summarization = sum(t["summarization"] for t in all_times) / len(all_times)
        avg_total = sum(t["total"] for t in all_times) / len(all_times)

        return {
            "average_preprocessing_time": avg_preprocessing,
            "average_ocr_time": avg_ocr,
            "average_summarization_time": avg_summarization,
            "average_total_time": avg_total,
            "estimated_time_for_400_files": avg_total * 400 / 60,  # in minutes
            "throughput_files_per_hour": 3600 / avg_total if avg_total > 0 else 0,
        }

    def identify_challenges(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify challenges from processing results."""
        challenges = []

        # Check for common issues
        low_confidence_count = 0
        empty_text_count = 0
        preprocessing_failures = 0

        for result in results:
            if not result["stages"]["preprocessing"]["success"]:
                preprocessing_failures += 1

            ocr_result = result["stages"]["ocr"]
            if ocr_result["success"]:
                if ocr_result.get("confidence", 0) < 0.7:
                    low_confidence_count += 1
                if not ocr_result["text"].strip():
                    empty_text_count += 1

        if preprocessing_failures > 0:
            challenges.append(
                f"Image preprocessing failed for {preprocessing_failures} files"
            )

        if low_confidence_count > 0:
            challenges.append(
                f"Low OCR confidence (<70%) for {low_confidence_count} files"
            )

        if empty_text_count > 0:
            challenges.append(f"No text extracted from {empty_text_count} files")

        if self.metrics["failed_summaries"] > 0:
            challenges.append(
                f"Summarization failed for {self.metrics['failed_summaries']} files"
            )

        return challenges

    def get_successful_examples(
        self, results: List[Dict[str, Any]], limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Get examples of successful processing."""
        successful = []

        for result in results:
            if result["stages"]["ocr"]["success"] and result["stages"].get(
                "summarization", {}
            ).get("success"):
                example = {
                    "file_name": result["file_name"],
                    "extracted_text": result["stages"]["ocr"]["text"][:200] + "...",
                    "summary": result["stages"]["summarization"]["summary"],
                    "category": result["stages"]["summarization"]["category"],
                    "confidence": result["stages"]["ocr"]["confidence"],
                    "processing_time": result["total_processing_time"],
                }
                successful.append(example)

                if len(successful) >= limit:
                    break

        return successful

    def get_failed_examples(
        self, results: List[Dict[str, Any]], limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Get examples of failed processing."""
        failed = []

        for result in results:
            if not result["stages"]["ocr"]["success"] or not result["stages"].get(
                "summarization", {}
            ).get("success"):
                example = {
                    "file_name": result["file_name"],
                    "ocr_error": result["stages"]["ocr"].get("error"),
                    "summary_error": result["stages"]
                    .get("summarization", {})
                    .get("error"),
                    "preprocessing_success": result["stages"]["preprocessing"][
                        "success"
                    ],
                }
                failed.append(example)

                if len(failed) >= limit:
                    break

        return failed

    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Based on success rates
        ocr_success_rate = self.metrics["successful_ocr"] / max(
            1, self.metrics["total_files"]
        )

        if ocr_success_rate < 0.8:
            recommendations.append(
                "Consider additional image preprocessing techniques for better OCR accuracy"
            )

        if self.metrics["failed_summaries"] > 0 and not self.summarizer:
            recommendations.append(
                "Configure Claude API credentials for text summarization capabilities"
            )

        # Based on performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics.get("average_ocr_time", 0) > 5:
            recommendations.append(
                "Consider using GPU-accelerated OCR for faster processing"
            )

        if (
            perf_metrics.get("estimated_time_for_400_files", 0) > 120
        ):  # More than 2 hours
            recommendations.append(
                "Implement parallel processing to reduce total processing time"
            )

        # Based on challenges
        challenges = self.identify_challenges(results)
        if any("Low OCR confidence" in c for c in challenges):
            recommendations.append(
                "Use Google Cloud Vision API for documents with low Tesseract confidence"
            )

        # General recommendations
        recommendations.extend(
            [
                "Implement caching to avoid reprocessing identical documents",
                "Set up monitoring dashboard for real-time processing status",
                "Create automated quality assurance checks for OCR accuracy",
                "Consider human-in-the-loop review for low-confidence extractions",
            ]
        )

        return recommendations

    def calculate_cost_estimates(self) -> Dict[str, Any]:
        """Calculate cost estimates for full collection processing."""
        # Based on our pipeline design from Task 17
        num_files = 400

        # OCR costs
        tesseract_cost = 0  # Free
        google_vision_cost = num_files * 0.0015  # $1.50 per 1000 images

        # AI summarization costs
        avg_tokens_per_doc = 2500  # Estimate
        claude_input_cost = (
            num_files * avg_tokens_per_doc * 3
        ) / 1_000_000  # $3 per million
        claude_output_cost = (num_files * 500 * 15) / 1_000_000  # $15 per million

        # Infrastructure costs (monthly estimates)
        compute_cost = 75  # Average of $50-100
        storage_cost = 7.5  # Average of $5-10

        # Time estimates
        perf_metrics = self.calculate_performance_metrics()
        processing_hours = perf_metrics.get("estimated_time_for_400_files", 240) / 60

        return {
            "one_time_processing": {
                "tesseract_ocr": tesseract_cost,
                "google_vision_ocr": google_vision_cost,
                "claude_api": claude_input_cost + claude_output_cost,
                "total": tesseract_cost + claude_input_cost + claude_output_cost,
            },
            "monthly_operational": {
                "compute": compute_cost,
                "storage": storage_cost,
                "total": compute_cost + storage_cost,
            },
            "time_estimates": {
                "hours_required": processing_hours,
                "days_required": processing_hours / 8,  # Assuming 8-hour days
            },
            "recommendation": "Use Tesseract for development and testing, Google Vision for production",
        }

    def save_report(self, report: Dict[str, Any]):
        """Save the comprehensive report."""
        report_path = Path("output/reports/poc_findings_report.json")
        report_path.write_text(json.dumps(report, indent=2))

        # Create a markdown version for readability
        md_report = self.create_markdown_report(report)
        md_path = Path("output/reports/poc_findings_report.md")
        md_path.write_text(md_report)

        logger.info(f"Report saved to: {report_path} and {md_path}")

    def create_markdown_report(self, report: Dict[str, Any]) -> str:
        """Create a markdown version of the report."""
        md = f"""# {report['title']}

Generated: {report['timestamp']}

## Executive Summary

- **Files Processed**: {report['summary']['total_files_processed']}
- **OCR Success Rate**: {report['summary']['ocr_success_rate']:.1%}
- **Summarization Success Rate**: {report['summary']['summary_success_rate']:.1%}

## Performance Metrics

- **Average Processing Time**: {report['performance_metrics']['average_total_time']:.2f} seconds/file
- **Estimated Time for 400 Files**: {report['performance_metrics']['estimated_time_for_400_files']:.1f} minutes
- **Throughput**: {report['performance_metrics']['throughput_files_per_hour']:.1f} files/hour

## Challenges and Limitations

"""
        for challenge in report["challenges_and_limitations"]:
            md += f"- {challenge}\n"

        md += "\n## Successful Examples\n\n"
        for example in report["successful_examples"]:
            md += f"### {example['file_name']}\n"
            md += f"- **Category**: {example['category']}\n"
            md += f"- **Confidence**: {example['confidence']:.2f}\n"
            md += f"- **Summary**: {example['summary']}\n\n"

        md += "## Recommendations\n\n"
        for rec in report["recommendations"]:
            md += f"- {rec}\n"

        md += "\n## Cost Estimates\n\n"
        md += f"### One-time Processing (400 files)\n"
        md += f"- Tesseract OCR: ${report['cost_estimates']['one_time_processing']['tesseract_ocr']:.2f}\n"
        md += f"- Claude API: ${report['cost_estimates']['one_time_processing']['claude_api']:.2f}\n"
        md += f"- **Total**: ${report['cost_estimates']['one_time_processing']['total']:.2f}\n\n"

        md += f"### Time Estimates\n"
        md += f"- Hours Required: {report['cost_estimates']['time_estimates']['hours_required']:.1f}\n"
        md += f"- Days Required: {report['cost_estimates']['time_estimates']['days_required']:.1f}\n"

        return md


def main():
    """Main function to run the proof of concept."""
    print("=== Hansman Syracuse Collection OCR Proof of Concept ===\n")

    # Initialize POC
    poc = HansmanOCRProofOfConcept(use_claude=True)

    # Get sample files
    sample_files = poc.get_sample_files(limit=10)

    if not sample_files:
        print("No sample files found. Please add files to data/hansman_samples/")
        return

    print(f"Found {len(sample_files)} files to process\n")

    # Process files
    results = poc.process_batch(sample_files)

    # Generate report
    report = poc.generate_report(results)

    # Save report
    poc.save_report(report)

    print("\n=== Processing Complete ===")
    print(f"OCR Success Rate: {report['summary']['ocr_success_rate']:.1%}")
    print(
        f"Average Processing Time: {report['performance_metrics']['average_total_time']:.2f} seconds/file"
    )
    print(f"Report saved to: output/reports/poc_findings_report.md")


if __name__ == "__main__":
    main()
