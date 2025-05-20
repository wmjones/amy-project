#!/usr/bin/env python3
"""Test the updated AI summarizer with better JSON parsing."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
from src.claude_integration.client_fixed import ClaudeClient


def test_updated_summarizer(file_name: str):
    """Test the updated AI summarizer."""

    # Initialize the AI summarizer with fixed client
    claude_client = ClaudeClient()
    ai_summarizer = AISummarizer(claude_client=claude_client)

    # Get the OCR text from the existing OCR file
    ocr_file = Path(
        f"/workspaces/amy-project/hansman_results/ocr_results/{Path(file_name).stem}_ocr.txt"
    )

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    print(f"OCR text loaded, length: {len(ocr_text)} characters")

    # Process through AI summarizer
    file_path = Path(f"/workspaces/amy-project/workspace/downloads/hansman/{file_name}")

    print(f"\nProcessing file through updated AI summarizer...")
    summary_result = ai_summarizer.summarize_document(
        file_path=file_path,
        ocr_text=ocr_text,
        additional_context={
            "collection": "Hansman Syracuse photo docs July 2015",
            "processing_date": "2025-05-19",
            "low_ocr_confidence": True,
            "ocr_confidence": 26.4,
        },
    )

    # Save the new summary
    new_summary_file = Path(
        f"/workspaces/amy-project/hansman_results/summaries/{Path(file_name).stem}_summary_final.json"
    )

    from dataclasses import asdict

    summary_data = asdict(summary_result)

    # Convert datetime objects to strings
    if "created_at" in summary_data and hasattr(
        summary_data["created_at"], "isoformat"
    ):
        summary_data["created_at"] = summary_data["created_at"].isoformat()

    with open(new_summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"\nSUCCESS! New summary saved to: {new_summary_file}")
    print(f"Summary: {summary_data.get('summary')[:300]}...")
    print(f"Category: {summary_data.get('category')}")
    print(f"Confidence score: {summary_data.get('confidence_score')}")
    print(f"Error message: {summary_data.get('error_message')}")

    # Show key entities
    print(f"\nKey entities: {summary_data.get('key_entities')}")
    print(f"Location references: {summary_data.get('location_references')}")
    print(f"Historical period: {summary_data.get('historical_period')}")
    print(f"Suggested folder: {summary_data.get('suggested_folder_path')}")


if __name__ == "__main__":
    # Test the problematic file
    test_updated_summarizer("100_4247.JPG")
