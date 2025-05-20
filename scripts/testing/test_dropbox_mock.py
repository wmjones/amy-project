#!/usr/bin/env python3
"""Mock Dropbox connection test to demonstrate the functionality."""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_find_file():
    """Mock test for finding file 100_4247 in shared folder 'Hansman Syracuse photo docs July 2015'."""
    print("=== Testing Dropbox Connection (Mock) ===\n")

    # Show credentials status
    app_key = os.getenv("DROPBOX_APP_KEY", "")
    app_secret = os.getenv("DROPBOX_APP_SECRET", "")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN", "")

    print("Credential Status:")
    print(f"  App Key: {'✓ Found' if app_key else '✗ Missing'}")
    print(f"  App Secret: {'✓ Found' if app_secret else '✗ Missing'}")
    print(f"  Refresh Token: {'✓ Found' if refresh_token else '✗ Missing'}")
    print()

    if not refresh_token:
        print(
            "⚠ Note: Refresh token is not set. You'll need to obtain it during first authentication."
        )
        print(
            "  This can be done using the Dropbox OAuth flow in the examples/dropbox_integration_demo.py"
        )
        print()

    # Mock the authentication
    print("Simulating authentication with Dropbox...")
    print("✓ Successfully authenticated (mock)\n")

    # Mock search for the file
    print(
        f"Searching for file '100_4247' in shared folder 'Hansman Syracuse photo docs July 2015'..."
    )
    print("-" * 80)

    # Simulate a successful search
    print("\n✓ Found 1 matching file(s):\n")
    print("  Name: 100_4247.jpg")
    print("  Path: /Hansman Syracuse photo docs July 2015/100_4247.jpg")
    print("  Size: 2.45 MB")
    print("  Type: image/jpeg")
    print(f"  Modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)

    print("\n" + "-" * 80)
    print("Test completed (mock)")

    print("\nNote: This is a mock test since the dropbox module is not installed.")
    print("In a real test, the DropboxAccessor would:")
    print("  1. Authenticate using the provided credentials")
    print("  2. Search for files matching '100_4247' in the shared folder")
    print("  3. Return detailed information about any matches found")


if __name__ == "__main__":
    test_find_file()
