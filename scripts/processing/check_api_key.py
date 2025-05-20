#!/usr/bin/env python3
"""Check API key availability."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API key available: {bool(api_key)}")
print(f"API key length: {len(api_key) if api_key else 0}")
print(f"API key prefix: {api_key[:10]}..." if api_key and len(api_key) > 10 else "Not available")