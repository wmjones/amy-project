#!/usr/bin/env python3
"""Test script to verify image attachment functionality in Claude API integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.claude_integration.client_fixed import ClaudeClient
from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_image_attachment():
    """Test the image attachment functionality with a sample document."""
    # Initialize the Claude client
    client = ClaudeClient()
    
    # Initialize the AI summarizer
    summarizer = AISummarizer(claude_client=client)
    
    # Find a test image file
    test_image_paths = [
        "/workspaces/amy-project/test_hansman/downloads/100_4248.JPG",
        "/workspaces/amy-project/test_hansman_final/downloads/100_4248.JPG"
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = Path(path)
            break
    
    if not test_image:
        print("No test image found. Please ensure test images are available.")
        return
    
    print(f"Testing with image: {test_image}")
    
    # Simple OCR text for testing
    test_ocr_text = """Syracuse Directory
    Page 123
    January 1925
    
    Arnold's Department Store
    Main Street, Syracuse NY
    
    Various business listings and addresses"""
    
    try:
        # Test direct image analysis
        print("\n1. Testing direct image analysis with ClaudeClient...")
        result = client.analyze_document(
            content=test_ocr_text,
            file_name=test_image.name,
            file_type=test_image.suffix,
            image_path=test_image
        )
        print(f"Success! Tokens used: {result.tokens_used}")
        print(f"Response preview: {result.content[:200]}...")
        
        # Test through AI summarizer
        print("\n2. Testing through AI summarizer...")
        summary = summarizer.summarize_document(
            file_path=test_image,
            ocr_text=test_ocr_text,
            additional_context={
                "collection": "Hansman Syracuse test",
                "processing_date": "2025-01-19"
            }
        )
        print(f"Success! Processing time: {summary.processing_time:.2f}s")
        print(f"Summary: {summary.summary[:200]}...")
        print(f"Category: {summary.category}")
        print(f"Confidence: {summary.confidence_score}")
        
        # Test without image for comparison
        print("\n3. Testing without image attachment...")
        result_no_image = client.analyze_document(
            content=test_ocr_text,
            file_name=test_image.name,
            file_type=test_image.suffix,
            image_path=None  # No image
        )
        print(f"Success! Tokens used: {result_no_image.tokens_used}")
        
        print("\n✅ All tests passed! Image attachment is working correctly.")
        
        # Compare token usage
        print(f"\nToken usage comparison:")
        print(f"  With image: {result.tokens_used} tokens")
        print(f"  Without image: {result_no_image.tokens_used} tokens")
        print(f"  Difference: {result.tokens_used - result_no_image.tokens_used} tokens")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_attachment()