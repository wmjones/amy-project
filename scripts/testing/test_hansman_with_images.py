#!/usr/bin/env python3
"""Test the updated Hansman processing pipeline with image attachments."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from src.claude_integration.client_fixed import ClaudeClient
from src.metadata_extraction.ai_summarizer_fixed import AISummarizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_hansman_pipeline():
    """Test the Hansman pipeline with image attachment support."""
    print("Testing Hansman Pipeline with Image Attachment Support")
    print("=" * 60)
    
    # Find test image
    test_images = [
        "/workspaces/amy-project/test_hansman/downloads/100_4248.JPG",
        "/workspaces/amy-project/test_hansman/downloads/100_4249.JPG",
        "/workspaces/amy-project/test_hansman/downloads/100_4250.JPG"
    ]
    
    existing_images = [Path(p) for p in test_images if Path(p).exists()]
    
    if not existing_images:
        print("No test images found.")
        return
    
    # Initialize components
    client = ClaudeClient()
    summarizer = AISummarizer(claude_client=client)
    
    # Find corresponding OCR results
    for image_path in existing_images[:1]:  # Test with first image
        print(f"\nProcessing: {image_path.name}")
        print("-" * 40)
        
        # Look for OCR text
        ocr_text_path = image_path.parent.parent / "ocr_results" / f"{image_path.stem}_ocr.txt"
        
        if ocr_text_path.exists():
            with open(ocr_text_path, 'r') as f:
                ocr_text = f.read()
            print(f"OCR text found: {len(ocr_text)} characters")
        else:
            ocr_text = "Sample OCR text for testing"
            print("Using sample OCR text")
        
        # Process with AI summarizer (with image)
        print("\nProcessing with image attachment...")
        try:
            summary = summarizer.summarize_document(
                file_path=image_path,
                ocr_text=ocr_text,
                additional_context={
                    "collection": "Hansman Syracuse photo docs July 2015",
                    "processing_date": "2025-01-19"
                }
            )
            
            print(f"✅ Success! Processing time: {summary.processing_time:.2f}s")
            print(f"\nResults:")
            print(f"  Summary: {summary.summary[:150]}...")
            print(f"  Category: {summary.category}")
            print(f"  Confidence: {summary.confidence_score:.2f}")
            print(f"  Historical Period: {summary.historical_period}")
            print(f"  Location References: {', '.join(summary.location_references[:3])}")
            print(f"  Key Entities: {len(summary.key_entities.get('people', [])) + len(summary.key_entities.get('locations', [])) + len(summary.key_entities.get('dates', []))} total")
            print(f"  Suggested Path: {summary.suggested_folder_path}")
            print(f"  Tokens Used: {summary.claude_metadata.get('tokens_used', 'N/A')}")
            
            # Save results
            output_path = image_path.parent / f"{image_path.stem}_enhanced_summary.json"
            import json
            from dataclasses import asdict
            
            with open(output_path, 'w') as f:
                json.dump(asdict(summary), f, indent=2, default=str)
            print(f"\n📁 Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete. The enhanced pipeline now includes original images")
    print("with OCR text for better document understanding and classification.")

if __name__ == "__main__":
    test_hansman_pipeline()