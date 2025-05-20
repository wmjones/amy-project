#!/usr/bin/env python3
"""Deep debug of AI processing issue."""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.claude_integration.client_fixed import ClaudeClient, AnalysisResult

# Enable maximum debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def test_simple_claude_call():
    """Test a simple Claude call to isolate the issue."""
    
    print("=== Test 1: Simple direct Claude call ===")
    
    try:
        client = ClaudeClient()
        
        # Very simple test
        result = client.analyze_document(
            content="This is a simple test document.",
            file_name="test.txt",
            file_type=".txt"
        )
        
        print(f"Success! Result: {result}")
        print(f"Content: {result.content[:200]}")
        print(f"Tokens: {result.tokens_used}")
        
    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()

def test_json_specific_prompt():
    """Test with a specific JSON prompt."""
    
    print("\n=== Test 2: JSON-specific prompt ===")
    
    try:
        client = ClaudeClient()
        
        # Force JSON response
        custom_prompt = '''Return ONLY valid JSON with no additional text:
{
    "summary": "Test summary",
    "category": "test"
}'''
        
        result = client.analyze_document(
            content="Test content",
            custom_prompt=custom_prompt,
            system_prompt="You are a JSON generator. Return only valid JSON."
        )
        
        print(f"Raw content: {repr(result.content)}")
        
        try:
            parsed = json.loads(result.content)
            print(f"Parsed successfully: {parsed}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content causing error: {repr(result.content)}")
            
    except Exception as e:
        print(f"Error in JSON test: {e}")
        import traceback
        traceback.print_exc()

def test_problematic_ocr():
    """Test with the actual problematic OCR text."""
    
    print("\n=== Test 3: Problematic OCR text ===")
    
    # Get the OCR text
    ocr_file = Path("/workspaces/amy-project/hansman_results/ocr_results/100_4247_ocr.txt")
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    print(f"OCR text length: {len(ocr_text)}")
    print(f"First 100 chars: {repr(ocr_text[:100])}")
    
    try:
        client = ClaudeClient()
        
        # Use a very simple system prompt
        simple_system = "Analyze this OCR text and return a JSON summary."
        
        # Use a minimal custom prompt
        custom_prompt = f"""Analyze this OCR text and return only this JSON structure:
{{
    "summary": "Brief description of the content",
    "category": "document type",
    "confidence_score": 0.5
}}

OCR Text: {ocr_text[:200]}"""
        
        result = client.analyze_document(
            content=ocr_text[:200],  # Limit to avoid overwhelming
            custom_prompt=custom_prompt,
            system_prompt=simple_system
        )
        
        print(f"\nRaw response: {repr(result.content[:500])}")
        
        # Try to find JSON in response
        import re
        json_match = re.search(r'\{[^}]+\}', result.content, re.DOTALL)
        if json_match:
            print(f"\nFound JSON pattern: {json_match.group(0)}")
            try:
                parsed = json.loads(json_match.group(0))
                print(f"Parsed: {parsed}")
            except json.JSONDecodeError as e:
                print(f"Parse error: {e}")
        else:
            print("No JSON pattern found in response")
            
    except Exception as e:
        print(f"Error in OCR test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_claude_call()
    test_json_specific_prompt()
    test_problematic_ocr()