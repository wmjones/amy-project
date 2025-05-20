#!/usr/bin/env python3
"""Simple test to verify Dropbox connection."""

import os
import dropbox
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_connection():
    """Test basic Dropbox connection."""
    app_key = os.getenv("DROPBOX_APP_KEY")
    app_secret = os.getenv("DROPBOX_APP_SECRET")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")

    print(f"App Key: {app_key}")
    print(f"App Secret: {app_secret[:5]}...")
    print(f"Refresh Token: {refresh_token[:10]}...")

    try:
        # Create Dropbox client
        print("\nCreating Dropbox client...")
        dbx = dropbox.Dropbox(
            app_key=app_key, app_secret=app_secret, oauth2_refresh_token=refresh_token
        )

        # Test the connection
        print("Getting account info...")
        account = dbx.users_get_current_account()
        print(f"✓ Success! Connected to account: {account.email}")

        # Try to search for the file
        print("\nSearching for file '100_4247'...")
        result = dbx.files_search_v2(query="100_4247")

        print(f"Found {len(result.matches)} results")
        for match in result.matches:
            metadata = match.metadata.get_metadata()
            if hasattr(metadata, "path_display"):
                print(f"  - {metadata.path_display}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")


if __name__ == "__main__":
    test_connection()
