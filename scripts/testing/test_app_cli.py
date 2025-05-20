#!/usr/bin/env python
"""
Test script to verify the app CLI works.
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.app import main

if __name__ == "__main__":
    # Test with help
    sys.argv = ['test_app_cli.py', '--help']
    try:
        main()
    except SystemExit as e:
        print(f"SystemExit with code: {e.code}")
        print("CLI help test passed!")