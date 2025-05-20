#!/usr/bin/env python3
"""Trace the exact error in AI processing."""

import sys
import json
import logging
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
from src.claude_integration.client_fixed import ClaudeClient


def trace_error():
    """Trace the exact error location."""

    print("=== Tracing Error in AI Summarizer ===")

    # Initialize components
    claude_client = ClaudeClient()
    ai_summarizer = AISummarizer(claude_client=claude_client)

    # Get OCR text
    ocr_file = Path(
        "/workspaces/amy-project/hansman_results/ocr_results/100_4247_ocr.txt"
    )
    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    # Try to call the AI summarizer directly
    try:
        file_path = Path(
            "/workspaces/amy-project/workspace/downloads/hansman/100_4247.JPG"
        )

        # Call the summarizer's internal methods to trace where it fails
        prompt = ai_summarizer._prepare_syracuse_prompt(ocr_text, file_path, {})
        print(f"\nPrompt prepared successfully")
        print(f"Prompt length: {len(prompt)}")
        print(f"Prompt preview: {prompt[:200]}...")

        # Try to get Claude response
        try:
            result = ai_summarizer.claude_client.analyze_document(
                content=ocr_text,
                file_name=str(file_path.name),
                file_type=file_path.suffix,
                custom_prompt=prompt,
                system_prompt=ai_summarizer.HANSMAN_SPECIFIC_PROMPT,
            )
            print(f"\nGot Claude response")
            print(f"Response type: {type(result)}")
            print(f"Response content length: {len(result.content)}")
            print(f"Response content preview: {result.content[:200]}...")

            # Try to parse the response
            try:
                parsed = ai_summarizer._parse_claude_response(result)
                print(f"\nSuccessfully parsed response: {type(parsed)}")
                print(f"Parsed keys: {list(parsed.keys())}")
            except Exception as parse_error:
                print(f"\nError parsing response: {parse_error}")
                print(f"Error type: {type(parse_error)}")
                print(f"Traceback:")
                traceback.print_exc()

                # Check the actual content
                print(f"\nActual response content:")
                print(repr(result.content))

        except Exception as claude_error:
            print(f"\nError getting Claude response: {claude_error}")
            traceback.print_exc()

    except Exception as e:
        print(f"\nError in AI summarizer: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    trace_error()
