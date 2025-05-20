#!/usr/bin/env python3
"""
Demo script for AI-driven text summarization and classification.
Tests the integration of OCR processing with Claude AI for the Hansman Syracuse collection.
"""

import sys
import os
from pathlib import Path
import json
import logging
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.file_access.ocr_processor import OCRProcessor
from src.metadata_extraction.ai_summarizer import AISummarizer
from src.metadata_extraction.ocr_ai_pipeline import OCRAIPipeline
from src.claude_integration.client import ClaudeClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_ai_summarization():
    """Demonstrate AI summarization on sample documents."""
    print("=" * 50)
    print("AI-Driven Summarization Demo")
    print("=" * 50)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key to run this demo.")
        return

    # Initialize components
    print("\n1. Initializing components...")

    try:
        # Initialize OCR processor
        ocr_processor = OCRProcessor()
        print("✓ OCR processor initialized")

        # Initialize Claude client
        claude_client = ClaudeClient()
        print("✓ Claude client initialized")

        # Initialize AI summarizer
        ai_summarizer = AISummarizer(claude_client=claude_client)
        print("✓ AI summarizer initialized")

        # Initialize pipeline
        pipeline = OCRAIPipeline(
            ocr_processor=ocr_processor,
            ai_summarizer=ai_summarizer,
            claude_client=claude_client,
        )
        print("✓ OCR + AI pipeline initialized")

    except Exception as e:
        print(f"\nError initializing components: {e}")
        return

    # Test with sample text
    print("\n2. Testing with sample Syracuse historical text...")

    sample_text = """
    Syracuse Salt Company
    Certificate of Stock

    This certifies that John H. Smith is the owner of fifty shares
    of the capital stock of the Syracuse Salt Company, incorporated
    under the laws of the State of New York.

    Dated this 15th day of March, 1892

    Located at: Salina Street, Syracuse, New York

    [Signed]
    William R. Johnson, President
    Robert E. Davis, Secretary
    """

    # Create temporary test file
    test_file = Path("./test_syracuse_document.txt")
    test_file.write_text(sample_text)

    try:
        # Test AI summarization directly
        print("\n3. Testing direct AI summarization...")

        summary = ai_summarizer.summarize_document(
            file_path=test_file,
            ocr_text=sample_text,
            ocr_confidence=0.95,
            additional_context={"source": "test_sample"},
        )

        print(f"\nDocument Summary:")
        print(f"Category: {summary.category}")
        print(f"Historical Period: {summary.historical_period}")
        print(f"Confidence: {summary.confidence_score:.2f}")
        print(f"Key Entities: {summary.key_entities}")
        print(f"Date References: {summary.date_references}")
        print(f"Location References: {summary.location_references}")
        print(f"Suggested Path: {summary.suggested_folder_path}")
        print(f"\nSummary Text: {summary.summary}")

        # Test pipeline processing
        print("\n4. Testing complete pipeline...")

        result = pipeline.process_file(test_file)

        if result.success:
            print(f"\n✓ Pipeline processing successful!")
            print(f"OCR Confidence: {result.ocr_result.confidence:.2f}")
            print(f"AI Classification: {result.ai_summary.category}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Cache Hit: {result.cache_hit}")
        else:
            print(f"\n✗ Pipeline error: {result.error_message}")

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

    # Test with actual Hansman files if available
    hansman_dir = Path("./hansman_organized")
    if hansman_dir.exists():
        print("\n5. Testing with actual Hansman Syracuse files...")

        # Find some test files
        test_files = []
        for ext in ["*.jpg", "*.png", "*.pdf"]:
            test_files.extend(list(hansman_dir.rglob(ext))[:3])

        if test_files:
            print(f"\nProcessing {len(test_files)} files from Hansman collection...")

            results = pipeline.process_batch(
                test_files[:3],  # Limit to 3 for demo
                progress_callback=lambda done, total: print(
                    f"Progress: {done}/{total}"
                ),
            )

            print("\nBatch Processing Results:")
            for i, result in enumerate(results):
                print(f"\nFile {i+1}: {result.file_path.name}")
                print(f"  Success: {result.success}")
                if result.success:
                    print(f"  Category: {result.ai_summary.category}")
                    print(f"  Period: {result.ai_summary.historical_period}")
                    print(f"  Confidence: {result.ai_summary.confidence_score:.2f}")
                    print(f"  Summary: {result.ai_summary.summary[:100]}...")
                else:
                    print(f"  Error: {result.error_message}")

            # Generate report
            report = pipeline.generate_pipeline_report(results)
            print("\nPipeline Report:")
            print(json.dumps(report, indent=2))

            # Export results
            export_path = pipeline.export_results(results, format="summary")
            print(f"\nResults exported to: {export_path}")
        else:
            print("\nNo test files found in Hansman collection.")
    else:
        print("\n5. Hansman collection directory not found. Skipping real file test.")

    # Demonstrate collection analysis
    print("\n6. Demonstrating collection analysis report...")

    # Create sample summaries for report
    from datetime import datetime

    sample_summaries = []

    categories = ["photo", "letter", "business_record", "certificate"]
    periods = ["salt_era", "canal_era", "industrial_boom", "modern"]
    locations = ["Armory Square", "Clinton Square", "Erie Canal", "Salina Street"]

    for i in range(10):
        summary = ai_summarizer.DocumentSummary(
            file_path=f"test_{i}.jpg",
            ocr_text="Sample text",
            summary=f"Test document {i}",
            category=categories[i % len(categories)],
            confidence_score=0.75 + (i % 3) * 0.1,
            key_entities={
                "people": [f"Person {i}"],
                "locations": [locations[i % len(locations)]],
            },
            date_references=[f"{1850 + i * 10}"],
            photo_subjects=[],
            location_references=[locations[i % len(locations)]],
            content_type="photo" if i % 2 == 0 else "document",
            historical_period=periods[i % len(periods)],
            classification_tags=[f"tag_{i}"],
            claude_metadata={},
            processing_time=2.5 + i * 0.1,
        )
        sample_summaries.append(summary)

    # Generate collection report
    collection_report = ai_summarizer.create_collection_report(sample_summaries)

    print("\nCollection Analysis Report:")
    print(json.dumps(collection_report, indent=2))

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)


def test_error_handling():
    """Test error handling in the pipeline."""
    print("\n" + "=" * 50)
    print("Testing Error Handling")
    print("=" * 50)

    # Initialize pipeline without API key to test error handling
    pipeline = OCRAIPipeline()

    # Test with non-existent file
    fake_file = Path("./non_existent_file.jpg")

    print("\n1. Testing with non-existent file...")
    result = pipeline.process_file(fake_file)

    print(f"Success: {result.success}")
    print(f"Error Message: {result.error_message}")

    # Test with invalid file
    invalid_file = Path("./test_invalid.txt")
    invalid_file.write_text("This is not an image file")

    print("\n2. Testing with invalid file type...")
    result = pipeline.process_file(invalid_file)

    print(f"Success: {result.success}")
    print(f"Error Message: {result.error_message}")

    # Clean up
    if invalid_file.exists():
        invalid_file.unlink()

    print("\nError handling test completed.")


if __name__ == "__main__":
    try:
        demo_ai_summarization()
        print("\n" + "=" * 50)
        test_error_handling()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        logger.exception("Demo failed with error")
