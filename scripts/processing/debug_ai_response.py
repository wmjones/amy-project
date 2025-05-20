#!/usr/bin/env python3
"""Debug the AI response for a problematic file."""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.claude_integration.client import ClaudeClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_claude_response(file_name: str):
    """Debug the Claude API response for a problematic file."""

    # Initialize Claude client
    claude_client = ClaudeClient()

    # Get the OCR text
    ocr_file = Path(
        f"/workspaces/amy-project/hansman_results/ocr_results/{Path(file_name).stem}_ocr.txt"
    )

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    print(f"OCR text loaded: {len(ocr_text)} characters")

    # Create a simple prompt
    prompt = (
        """You are analyzing OCR text from a historical document. Return valid JSON only:

{
    "summary": "Brief summary of the content",
    "category": "document type",
    "confidence_score": 0.5
}

OCR Text:
"""
        + ocr_text[:500]
    )  # Limit text to avoid overwhelming the API

    try:
        # Call Claude directly
        print("\nCalling Claude API...")
        result = claude_client.analyze_document(
            content=ocr_text[:500],
            file_name=file_name,
            file_type=".jpg",
            custom_prompt=prompt,
        )

        print(f"\nClaude response model: {result.model}")
        print(f"Tokens used: {result.tokens_used}")
        print(f"Content type: {type(result.content)}")
        print(f"Content length: {len(result.content)}")
        print(f"\nRaw content:")
        print(repr(result.content[:500]))

        # Try to parse as JSON
        try:
            parsed = json.loads(result.content)
            print("\nSuccessfully parsed as JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nJSON parse error: {e}")
            print(f"Error at position: {e.pos}")
            print(f"Error line/column: {e.lineno}:{e.colno}")

            # Show the problematic part
            if e.pos:
                start = max(0, e.pos - 50)
                end = min(len(result.content), e.pos + 50)
                print(f"\nProblematic section [{start}:{end}]:")
                print(repr(result.content[start:end]))

    except Exception as e:
        print(f"\nError calling Claude: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_claude_response("100_4247.JPG")
