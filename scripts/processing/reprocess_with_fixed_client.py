#!/usr/bin/env python3
"""Reprocess file with fixed Claude client."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
from src.claude_integration.client_fixed import ClaudeClient

def reprocess_file(file_name: str):
    """Reprocess a single file through the AI summarizer with fixed client."""
    
    # Initialize the AI summarizer with fixed client
    claude_client = ClaudeClient()
    ai_summarizer = AISummarizer(claude_client=claude_client)
    
    # Get the OCR text from the existing OCR file
    ocr_file = Path(f"/workspaces/amy-project/hansman_results/ocr_results/{Path(file_name).stem}_ocr.txt")
    
    if not ocr_file.exists():
        print(f"OCR file not found: {ocr_file}")
        return
    
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    print(f"OCR text loaded, length: {len(ocr_text)} characters")
    
    # Process through AI summarizer
    file_path = Path(f"/workspaces/amy-project/workspace/downloads/hansman/{file_name}")
    
    print(f"\nProcessing file through AI summarizer with fixed client...")
    summary_result = ai_summarizer.summarize_document(
        file_path=file_path,
        ocr_text=ocr_text,
        additional_context={
            'collection': 'Hansman Syracuse photo docs July 2015',
            'processing_date': '2025-05-19',
            'low_ocr_confidence': True,
            'ocr_confidence': 26.4
        }
    )
    
    # Save the new summary
    new_summary_file = Path(f"/workspaces/amy-project/hansman_results/summaries/{Path(file_name).stem}_summary_fixed_client.json")
    
    from dataclasses import asdict
    summary_data = asdict(summary_result)
    
    # Convert datetime objects to strings
    if 'created_at' in summary_data and hasattr(summary_data['created_at'], 'isoformat'):
        summary_data['created_at'] = summary_data['created_at'].isoformat()
    
    summary_data['processed_at'] = '2025-05-19T15:35:00'
    
    with open(new_summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\nNew summary saved to: {new_summary_file}")
    print(f"Summary: {summary_data.get('summary')[:200]}...")
    print(f"Category: {summary_data.get('category')}")
    print(f"Error message: {summary_data.get('error_message')}")
    print(f"Confidence score: {summary_data.get('confidence_score')}")

if __name__ == "__main__":
    # Reprocess the problematic file
    reprocess_file("100_4247.JPG")