#!/usr/bin/env python3
"""Diagnose Dropbox connection issues."""

import os
import dropbox
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()


def diagnose_connection():
    """Diagnose Dropbox connection issues."""
    print("=== Dropbox Connection Diagnostics ===\n")

    # Get credentials
    app_key = os.getenv("DROPBOX_APP_KEY")
    app_secret = os.getenv("DROPBOX_APP_SECRET")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")

    print("1. Checking environment variables:")
    print(f"   DROPBOX_APP_KEY: {'✓ Found' if app_key else '✗ Missing'} ({app_key})")
    print(
        f"   DROPBOX_APP_SECRET: {'✓ Found' if app_secret else '✗ Missing'} ({app_secret[:5]}...)"
    )
    print(
        f"   DROPBOX_REFRESH_TOKEN: {'✓ Found' if refresh_token else '✗ Missing'} ({refresh_token[:20]}...)"
    )

    if not all([app_key, app_secret, refresh_token]):
        print("\n✗ Missing required credentials")
        return

    print("\n2. Testing token format:")
    if refresh_token.startswith("sl."):
        print("   ✓ Token has correct prefix (sl.)")
    else:
        print("   ✗ Token doesn't have expected prefix")

    print("\n3. Direct API test:")
    try:
        # Test the refresh token directly
        response = requests.post(
            "https://api.dropboxapi.com/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": app_key,
                "client_secret": app_secret,
            },
        )

        print(f"   Response status: {response.status_code}")
        print(f"   Response: {response.json()}")

        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            print(f"\n   ✓ Successfully obtained access token")

            # Try using the access token
            print("\n4. Testing with access token:")
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            user_response = requests.post(
                "https://api.dropboxapi.com/2/users/get_current_account",
                headers=headers,
            )

            print(f"   Response status: {user_response.status_code}")
            if user_response.status_code == 200:
                user_data = user_response.json()
                print(f"   ✓ Connected to account: {user_data.get('email')}")

                # Try searching for file
                print("\n5. Searching for file '100_4247':")
                search_response = requests.post(
                    "https://api.dropboxapi.com/2/files/search_v2",
                    headers=headers,
                    json={
                        "query": "100_4247",
                        "options": {"path": "", "max_results": 10},
                    },
                )

                if search_response.status_code == 200:
                    search_data = search_response.json()
                    matches = search_data.get("matches", [])
                    print(f"   Found {len(matches)} matches")
                    for match in matches:
                        metadata = match.get("metadata", {}).get("metadata", {})
                        path = metadata.get("path_display", "")
                        print(f"   - {path}")
                        if "Hansman Syracuse photo docs July 2015" in path:
                            print(f"     ✓ Found in target folder!")
                else:
                    print(f"   Search failed: {search_response.status_code}")
                    print(f"   Error: {search_response.text}")
            else:
                print(f"   ✗ Failed to get user info: {user_response.text}")
        else:
            print(f"   ✗ Failed to refresh token: {response.text}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n6. OAuth URL for re-authentication:")
    print(
        f"   https://www.dropbox.com/oauth2/authorize?response_type=code&client_id={app_key}&token_access_type=offline"
    )


if __name__ == "__main__":
    diagnose_connection()
