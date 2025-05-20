#!/usr/bin/env python3
"""Debug AI summarizer issues."""

import os
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.metadata_extraction.ai_summarizer import AISummarizer
from src.claude_integration.client import ClaudeClient


def test_summarizer():
    """Test the AI summarizer with debugging."""
    # Initialize clients
    claude_client = ClaudeClient()
    summarizer = AISummarizer(claude_client)

    # Test with a simple text
    test_path = Path("test_document.jpg")
    test_ocr_text = "January 9, 1964"

    try:
        # Create custom prompt to ensure JSON response
        custom_prompt = """
        Analyze this OCR text and return ONLY a JSON object:

        OCR Text: January 9, 1964

        Return JSON with this structure:
        {
            "summary": "Brief summary",
            "category": "document type",
            "confidence_score": 0.8,
            "key_entities": {"people": [], "organizations": [], "locations": []},
            "date_references": ["1964"],
            "photo_subjects": [],
            "location_references": [],
            "content_type": "document",
            "historical_period": "1960s",
            "classification_tags": ["date", "letter"],
            "suggested_folder_path": "Hansman_Syracuse/Documents/1960s",
            "quality_indicators": {"text_clarity": 0.9, "historical_value": 0.5}
        }
        """

        # Test the analysis
        result = summarizer.claude_client.analyze_document(
            content=test_ocr_text,
            file_name="test_document.jpg",
            file_type=".jpg",
            custom_prompt=custom_prompt,
            system_prompt="Return only valid JSON. No explanation or text before or after.",
        )

        print("Raw Claude response:")
        print(result.content)
        print("\nResponse type:", type(result.content))

        # Try parsing
        try:
            parsed = json.loads(result.content)
            print("\nSuccessfully parsed JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_summarizer()
