#!/usr/bin/env python3
"""Check the status of Hansman processing."""

import json
from pathlib import Path

def check_status():
    # Load state file
    state_file = Path("hansman_full_processing/.state/processing_state.json")
    
    if not state_file.exists():
        print("No processing state found")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    print("Hansman Processing Status")
    print("=" * 30)
    print(f"Downloaded files: {len(state['downloaded_files'])}")
    print(f"OCR processed: {len(state['ocr_processed'])}")
    print(f"AI processed: {len(state['ai_processed'])}")
    print(f"Organized files: {len(state['organized_files'])}")
    print(f"Errors: {len(state['errors'])}")
    print(f"Last run: {state['last_run']}")
    
    # Check actual files
    downloads_dir = Path("hansman_full_processing/downloads")
    ocr_dir = Path("hansman_full_processing/ocr_results")
    summaries_dir = Path("hansman_full_processing/summaries")
    
    if downloads_dir.exists():
        actual_downloads = len(list(downloads_dir.glob("*.JPG")))
        print(f"\nActual downloaded files: {actual_downloads}")
    
    if ocr_dir.exists():
        actual_ocr = len(list(ocr_dir.glob("*_ocr.json")))
        print(f"Actual OCR files: {actual_ocr}")
    
    if summaries_dir.exists():
        actual_summaries = len(list(summaries_dir.glob("*_summary.json")))
        print(f"Actual AI summaries: {actual_summaries}")
    
    # Show recent errors
    if state['errors']:
        print("\nRecent errors:")
        for error in state['errors'][-5:]:
            print(f"- {error['stage']}: {error['file']} - {error['error']}")

if __name__ == "__main__":
    check_status()