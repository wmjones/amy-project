#!/usr/bin/env python3
"""Test the AI summarizer directly to debug the issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.DEBUG)

from src.metadata_extraction.ai_summarizer import AISummarizer
from src.claude_integration.client import ClaudeClient

# Test with a small sample
test_text = """
FUND FOR ADULT EDUCATION
1225 19th Street, N.W.
Washington, D.C. 20036

November 20, 1964

Mr. Eugene I. Johnson
Executive Director
Adult Education Association, USA

Dear Gene:

This letter is to confirm our discussion regarding the upcoming conference.

Sincerely,
Test Author
"""

print("Testing AI summarizer...")

try:
    client = ClaudeClient()
    summarizer = AISummarizer(claude_client=client)
    
    result = summarizer.summarize_document(
        file_path=Path("test.jpg"),
        ocr_text=test_text,
        ocr_confidence=0.8
    )
    
    print(f"Success! Summary: {result.summary}")
    print(f"Category: {result.category}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()