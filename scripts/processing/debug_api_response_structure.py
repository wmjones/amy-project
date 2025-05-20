#!/usr/bin/env python3
"""Debug the Claude API response structure."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import anthropic
import os

def debug_api_structure():
    """Debug the actual API response structure."""
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Make a simple test call
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": "Return a simple JSON: {\"test\": \"hello\"}"
        }]
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response attributes: {dir(response)}")
    print(f"\nUsage type: {type(response.usage)}")
    print(f"Usage attributes: {dir(response.usage)}")
    print(f"\nUsage values:")
    print(f"  input_tokens: {response.usage.input_tokens}")
    print(f"  output_tokens: {response.usage.output_tokens}")
    
    # Check if total_tokens exists
    if hasattr(response.usage, 'total_tokens'):
        print(f"  total_tokens: {response.usage.total_tokens}")
    else:
        print(f"  total_tokens: NOT FOUND - calculating from input + output")
        print(f"  calculated_total: {response.usage.input_tokens + response.usage.output_tokens}")
    
    print(f"\nContent: {response.content[0].text}")

if __name__ == "__main__":
    debug_api_structure()