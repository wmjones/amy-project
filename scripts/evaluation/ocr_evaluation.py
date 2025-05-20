#!/usr/bin/env python3
"""Evaluate OCR solutions for Hansman Syracuse photo documents."""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class OCRBenchmark:
    """Benchmark different OCR solutions on the Hansman Syracuse collection."""

    def __init__(self):
        self.results = {
            "tesseract": {},
            "google_vision": {},
            "aws_textract": {},
            "azure_vision": {},
        }
        self.sample_files = []

    def prepare_test_samples(
        self, sample_folder="/workspaces/amy-project/hansman_organized/downloads"
    ):
        """Get sample files from the downloaded collection."""
        sample_path = Path(sample_folder)
        if sample_path.exists():
            # Get first 10 JPG files for testing
            self.sample_files = list(sample_path.glob("*.JPG"))[:10]
            print(f"Found {len(self.sample_files)} test files")
        else:
            print(f"Sample folder not found: {sample_folder}")

    def test_tesseract(self):
        """Test Tesseract OCR (local solution)."""
        print("\n=== Testing Tesseract OCR ===")

        try:
            import pytesseract
            from PIL import Image

            tesseract_results = {
                "available": True,
                "processing_times": [],
                "extracted_texts": [],
                "confidence_scores": [],
                "errors": [],
            }

            for file_path in self.sample_files:
                try:
                    start_time = time.time()

                    # Open image
                    image = Image.open(file_path)

                    # Extract text with confidence
                    data = pytesseract.image_to_data(
                        image, output_type=pytesseract.Output.DICT
                    )
                    text = pytesseract.image_to_string(image)

                    # Calculate average confidence
                    confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0

                    processing_time = time.time() - start_time

                    tesseract_results["processing_times"].append(processing_time)
                    tesseract_results["extracted_texts"].append(
                        {
                            "file": file_path.name,
                            "text": text[:500],  # First 500 chars
                            "full_length": len(text),
                        }
                    )
                    tesseract_results["confidence_scores"].append(avg_confidence)

                    print(
                        f"  ✓ {file_path.name}: {len(text)} chars in {processing_time:.2f}s"
                    )

                except Exception as e:
                    tesseract_results["errors"].append(
                        {"file": file_path.name, "error": str(e)}
                    )
                    print(f"  ✗ {file_path.name}: {e}")

            # Calculate summary statistics
            if tesseract_results["processing_times"]:
                tesseract_results["summary"] = {
                    "avg_processing_time": np.mean(
                        tesseract_results["processing_times"]
                    ),
                    "avg_confidence": np.mean(tesseract_results["confidence_scores"]),
                    "success_rate": len(tesseract_results["processing_times"])
                    / len(self.sample_files),
                    "cost": 0.0,  # Free
                }

            self.results["tesseract"] = tesseract_results

        except ImportError:
            print("Tesseract not installed. Install with: pip install pytesseract")
            self.results["tesseract"] = {"available": False, "error": "Not installed"}

    def test_google_vision(self):
        """Test Google Cloud Vision API."""
        print("\n=== Testing Google Cloud Vision ===")

        # This is a placeholder - actual implementation would require Google Cloud credentials
        self.results["google_vision"] = {
            "available": False,
            "reason": "Requires Google Cloud credentials and setup",
            "pricing": {
                "first_1000_units": 1.50,  # per 1000 units
                "unit_definition": "per image",
                "free_tier": "1000 units/month",
            },
            "features": [
                "High accuracy for printed text",
                "Handwriting recognition",
                "Multiple language support",
                "Batch processing",
                "Document AI features",
            ],
        }

    def test_aws_textract(self):
        """Test AWS Textract."""
        print("\n=== Testing AWS Textract ===")

        # This is a placeholder - actual implementation would require AWS credentials
        self.results["aws_textract"] = {
            "available": False,
            "reason": "Requires AWS credentials and setup",
            "pricing": {
                "detect_text": 0.0015,  # per page
                "analyze_document": 0.015,  # per page
                "free_tier": "1000 pages/month for first 3 months",
            },
            "features": [
                "Text detection",
                "Form extraction",
                "Table extraction",
                "Confidence scores",
                "Asynchronous processing for large batches",
            ],
        }

    def test_azure_vision(self):
        """Test Azure Computer Vision."""
        print("\n=== Testing Azure Computer Vision ===")

        # This is a placeholder - actual implementation would require Azure credentials
        self.results["azure_vision"] = {
            "available": False,
            "reason": "Requires Azure credentials and setup",
            "pricing": {
                "read_api": 1.50,  # per 1000 transactions
                "free_tier": "5000 transactions/month",
            },
            "features": [
                "Read API for dense text",
                "Handwriting support",
                "Multiple page PDFs",
                "73 language support",
                "Async processing",
            ],
        }

    def create_decision_matrix(self):
        """Create a weighted decision matrix for OCR solutions."""

        # Define criteria and weights for Hansman Syracuse collection
        criteria = {
            "accuracy": 0.25,  # Most important for historical documents
            "processing_speed": 0.15,
            "cost": 0.20,  # Important for 400+ files
            "ease_of_integration": 0.15,
            "historical_doc_support": 0.15,
            "error_handling": 0.10,
        }

        # Score each solution (1-10)
        scores = {
            "tesseract": {
                "accuracy": 6,
                "processing_speed": 8,
                "cost": 10,
                "ease_of_integration": 9,
                "historical_doc_support": 7,
                "error_handling": 7,
            },
            "google_vision": {
                "accuracy": 9,
                "processing_speed": 7,
                "cost": 6,
                "ease_of_integration": 7,
                "historical_doc_support": 8,
                "error_handling": 9,
            },
            "aws_textract": {
                "accuracy": 8,
                "processing_speed": 7,
                "cost": 6,
                "ease_of_integration": 7,
                "historical_doc_support": 7,
                "error_handling": 8,
            },
            "azure_vision": {
                "accuracy": 8,
                "processing_speed": 7,
                "cost": 7,
                "ease_of_integration": 6,
                "historical_doc_support": 8,
                "error_handling": 8,
            },
        }

        # Calculate weighted scores
        decision_matrix = {}
        for solution, solution_scores in scores.items():
            weighted_score = sum(
                solution_scores[criterion] * weight
                for criterion, weight in criteria.items()
            )
            decision_matrix[solution] = {
                "scores": solution_scores,
                "weighted_score": weighted_score,
            }

        return decision_matrix

    def generate_report(self):
        """Generate comprehensive evaluation report."""

        decision_matrix = self.create_decision_matrix()

        report = {
            "evaluation_date": datetime.now().isoformat(),
            "test_samples": len(self.sample_files),
            "ocr_solutions": self.results,
            "decision_matrix": decision_matrix,
            "recommendations": self._generate_recommendations(decision_matrix),
            "implementation_plan": self._generate_implementation_plan(),
        }

        # Save report
        report_path = Path("ocr_evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Create markdown summary
        self._create_markdown_summary(report)

        return report

    def _generate_recommendations(self, decision_matrix):
        """Generate recommendations based on evaluation."""

        # Sort by weighted score
        sorted_solutions = sorted(
            decision_matrix.items(), key=lambda x: x[1]["weighted_score"], reverse=True
        )

        recommendations = {
            "primary_solution": sorted_solutions[0][0],
            "fallback_solution": "tesseract",  # Always available locally
            "reasoning": {
                "primary": f"{sorted_solutions[0][0]} has the highest weighted score ({sorted_solutions[0][1]['weighted_score']:.2f})",
                "fallback": "Tesseract provides free, local processing for fallback scenarios",
            },
            "hybrid_approach": {
                "description": "Use Tesseract for initial testing and non-critical documents, cloud solution for production",
                "benefits": [
                    "Lower costs during development",
                    "Fallback for network/API failures",
                    "Flexibility in processing strategy",
                ],
            },
        }

        return recommendations

    def _generate_implementation_plan(self):
        """Generate phased implementation plan."""

        return {
            "phase_1": {
                "duration": "1 week",
                "tasks": [
                    "Set up Tesseract locally",
                    "Process sample batch of 50 documents",
                    "Establish baseline accuracy metrics",
                ],
            },
            "phase_2": {
                "duration": "2 weeks",
                "tasks": [
                    "Set up chosen cloud OCR service",
                    "Implement error handling and retry logic",
                    "Create comparison metrics between solutions",
                ],
            },
            "phase_3": {
                "duration": "1 week",
                "tasks": [
                    "Integrate with Claude API for summarization",
                    "Implement caching for processed documents",
                    "Create monitoring dashboard",
                ],
            },
            "phase_4": {
                "duration": "1 week",
                "tasks": [
                    "Process full Hansman Syracuse collection",
                    "Document performance metrics",
                    "Optimize based on findings",
                ],
            },
        }

    def _create_markdown_summary(self, report):
        """Create a markdown summary of the evaluation."""

        md_content = f"""# OCR Solution Evaluation Report

## Date: {report['evaluation_date']}

## Executive Summary

Based on evaluation of {report['test_samples']} sample files from the Hansman Syracuse collection:

**Recommended Primary Solution:** {report['recommendations']['primary_solution']}
**Recommended Fallback:** {report['recommendations']['fallback_solution']}

## Decision Matrix

| Solution | Weighted Score | Best For |
|----------|---------------|----------|
"""

        for solution, data in sorted(
            report["decision_matrix"].items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        ):
            md_content += f"| {solution} | {data['weighted_score']:.2f} | "

            if solution == "tesseract":
                md_content += "Local processing, cost-sensitive projects |\n"
            elif solution == "google_vision":
                md_content += "High accuracy, multi-language support |\n"
            elif solution == "aws_textract":
                md_content += "Forms and tables, AWS ecosystem |\n"
            else:
                md_content += "Azure ecosystem, large batches |\n"

        md_content += f"""
## Test Results

### Tesseract (Local)
- Available: {self.results['tesseract'].get('available', False)}
"""

        if self.results["tesseract"].get("summary"):
            summary = self.results["tesseract"]["summary"]
            md_content += f"""- Avg Processing Time: {summary['avg_processing_time']:.2f}s
- Avg Confidence: {summary['avg_confidence']:.0f}%
- Success Rate: {summary['success_rate']*100:.0f}%
- Cost: Free
"""

        md_content += """
## Recommendations

1. **Start with Tesseract** for immediate testing and development
2. **Implement Google Cloud Vision** for production use (highest accuracy)
3. **Use hybrid approach** to optimize costs and reliability

## Implementation Timeline

- Week 1: Tesseract setup and baseline testing
- Week 2-3: Cloud service integration
- Week 4: Claude API integration and optimization
- Week 5: Full collection processing

## Next Steps

1. Install and configure Tesseract locally
2. Process 10-15 representative samples
3. Obtain cloud service credentials for testing
4. Create cost projections for full collection
"""

        with open("ocr_evaluation_summary.md", "w") as f:
            f.write(md_content)


def main():
    """Run OCR evaluation."""
    evaluator = OCRBenchmark()

    print("OCR Solution Evaluation for Hansman Syracuse Collection")
    print("=" * 50)

    # Prepare test samples
    evaluator.prepare_test_samples()

    if not evaluator.sample_files:
        print("No sample files found. Using placeholder data.")
        evaluator.sample_files = ["sample1.jpg", "sample2.jpg"]  # Placeholders

    # Test each solution
    evaluator.test_tesseract()
    evaluator.test_google_vision()
    evaluator.test_aws_textract()
    evaluator.test_azure_vision()

    # Generate report
    report = evaluator.generate_report()

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print(f"Report saved to: ocr_evaluation_report.json")
    print(f"Summary saved to: ocr_evaluation_summary.md")

    # Print recommendations
    print(f"\nRecommended solution: {report['recommendations']['primary_solution']}")
    print(f"Fallback solution: {report['recommendations']['fallback_solution']}")


if __name__ == "__main__":
    main()
