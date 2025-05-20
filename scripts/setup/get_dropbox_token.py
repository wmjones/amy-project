#!/usr/bin/env python3
"""Get Dropbox refresh token through OAuth flow."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import Dropbox
import dropbox

def get_refresh_token():
    """Get refresh token through OAuth flow."""
    print("=== Dropbox OAuth Authentication ===\n")
    
    # Get app credentials from environment
    app_key = os.getenv('DROPBOX_APP_KEY')
    app_secret = os.getenv('DROPBOX_APP_SECRET')
    
    if not all([app_key, app_secret]):
        print("Error: DROPBOX_APP_KEY and DROPBOX_APP_SECRET must be set in .env")
        return
    
    print(f"Using App Key: {app_key}")
    print(f"Using App Secret: {app_secret[:5]}...\n")
    
    # Create OAuth flow
    auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        app_key, 
        app_secret,
        token_access_type='offline'  # Request refresh token
    )
    
    auth_url = auth_flow.start()
    print("1. Go to this URL (you may need to be in a browser where you're logged into Dropbox):")
    print(f"   {auth_url}\n")
    print("2. Click 'Allow' (you might have to log in first)")
    print("3. Copy the authorization code.")
    print("\nNote: Make sure your app has the necessary permissions for file access.")
    
    auth_code = input("\nEnter the authorization code here: ").strip()
    
    try:
        oauth_result = auth_flow.finish(auth_code)
        refresh_token = oauth_result.refresh_token
        access_token = oauth_result.access_token
        
        print(f"\n✓ Successfully authenticated!")
        print(f"\nRefresh token: {refresh_token}")
        print(f"\nAdd this to your .env file:")
        print(f"DROPBOX_REFRESH_TOKEN={refresh_token}")
        
        # Test the connection
        dbx = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=refresh_token
        )
        
        account = dbx.users_get_current_account()
        print(f"\n✓ Connected to Dropbox account: {account.email}")
        
    except Exception as e:
        print(f"\n✗ Error during OAuth flow: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    get_refresh_token()