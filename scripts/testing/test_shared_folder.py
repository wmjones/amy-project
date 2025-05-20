#!/usr/bin/env python3
"""Test Dropbox shared folder access."""

import os
import dropbox
from dropbox.files import ListFolderError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_shared_folder_access():
    """Test accessing shared folder and finding file."""
    app_key = os.getenv('DROPBOX_APP_KEY')
    app_secret = os.getenv('DROPBOX_APP_SECRET')
    refresh_token = os.getenv('DROPBOX_REFRESH_TOKEN')
    
    print("=== Testing Dropbox Shared Folder Access ===\n")
    
    # Try different authentication methods
    try:
        # Use refresh token with app credentials
        print("Attempting connection with refresh token and app credentials...")
        dbx = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=refresh_token
        )
        
        # Test the connection
        account = dbx.users_get_current_account()
        print(f"✓ Connected to account: {account.email}\n")
        
        # Try to search for the file
        print("Searching for file '100_4247'...")
        try:
            # Use the V2 search API
            result = dbx.files_search_v2(query="100_4247")
            
            print(f"Found {len(result.matches)} results:")
            for match in result.matches:
                metadata = match.metadata.get_metadata()
                if hasattr(metadata, 'path_display'):
                    print(f"  - {metadata.path_display}")
                    if "Hansman Syracuse photo docs July 2015" in metadata.path_display:
                        print(f"    ✓ Found in target folder!")
                        print(f"    Size: {metadata.size / (1024*1024):.2f} MB")
                        print(f"    Modified: {metadata.server_modified}")
        
        except Exception as e:
            print(f"Search error: {e}")
        
        # Try listing shared folders
        print("\nListing shared folders...")
        try:
            shared_folders = dbx.sharing_list_shared_links()
            print(f"Found {len(shared_folders.links)} shared links")
            for link in shared_folders.links[:5]:  # Show first 5
                print(f"  - {link.path}")
        except Exception as e:
            print(f"Error listing shared folders: {e}")
        
    except dropbox.exceptions.AuthError as e:
        print(f"Authentication error: {e}")
        print("\nTrying with app key and secret...")
        
        try:
            # Try with app credentials
            dbx = dropbox.Dropbox(
                app_key=app_key,
                app_secret=app_secret,
                oauth2_refresh_token=refresh_token
            )
            
            account = dbx.users_get_current_account()
            print(f"✓ Connected with app credentials: {account.email}")
            
        except Exception as e2:
            print(f"Still failed: {e2}")
            print("\nThe refresh token may have expired or been revoked.")
            print("You may need to re-authenticate using get_dropbox_token.py")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_shared_folder_access()