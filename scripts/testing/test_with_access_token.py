#!/usr/bin/env python3
"""Test Dropbox connection using access token."""

import os
import dropbox
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_with_access_token():
    """Test finding file using access token."""
    print("=== Testing Dropbox with Access Token ===\n")
    
    # Get access token
    access_token = os.getenv('DROPBOX_ACCESS_TOKEN')
    
    if not access_token:
        print("Please add DROPBOX_ACCESS_TOKEN to your .env file")
        print("\nTo get an access token:")
        print("1. Go to https://www.dropbox.com/developers/apps")
        print("2. Select your app (or create a new one)")
        print("3. In the Settings tab, click 'Generate' under 'Generated access token'")
        print("4. Copy the token and add to .env as:")
        print("   DROPBOX_ACCESS_TOKEN=your_token_here")
        return
    
    try:
        # Create Dropbox client with access token
        print("Connecting to Dropbox...")
        dbx = dropbox.Dropbox(access_token)
        
        # Test the connection
        account = dbx.users_get_current_account()
        print(f"✓ Connected to account: {account.email}\n")
        
        # Search for the file
        print("Searching for file '100_4247'...")
        result = dbx.files_search_v2(query="100_4247")
        
        found_in_target = False
        print(f"\nFound {len(result.matches)} matching files:")
        
        for match in result.matches:
            metadata = match.metadata.get_metadata()
            if hasattr(metadata, 'path_display'):
                print(f"\nFile: {metadata.name}")
                print(f"Path: {metadata.path_display}")
                print(f"Size: {metadata.size / (1024*1024):.2f} MB")
                print(f"Modified: {metadata.server_modified}")
                
                # Check if it's in the target folder
                if "Hansman Syracuse photo docs July 2015" in metadata.path_display:
                    print("✓ This is in the target shared folder!")
                    found_in_target = True
        
        if not found_in_target and result.matches:
            print("\n⚠ Files found but not in 'Hansman Syracuse photo docs July 2015' folder")
            print("The file might be in a different location.")
        elif not result.matches:
            print("\n✗ No files matching '100_4247' found")
            
            # Try listing the shared folder directly
            print("\nAttempting to list the shared folder...")
            try:
                # Try different path formats
                paths_to_try = [
                    "/Hansman Syracuse photo docs July 2015",
                    "Hansman Syracuse photo docs July 2015",
                    "/Shared/Hansman Syracuse photo docs July 2015"
                ]
                
                for path in paths_to_try:
                    try:
                        print(f"\nTrying path: {path}")
                        result = dbx.files_list_folder(path)
                        print(f"✓ Found folder! Contains {len(result.entries)} items")
                        
                        # Look for our file
                        for entry in result.entries:
                            if "100_4247" in entry.name:
                                print(f"✓ Found file: {entry.name}")
                                break
                        break
                    except Exception as e:
                        print(f"✗ {type(e).__name__}: {e}")
                        
            except Exception as e:
                print(f"Error accessing shared folder: {e}")
        
    except dropbox.exceptions.AuthError as e:
        print(f"\n✗ Authentication error: {e}")
        print("\nPossible causes:")
        print("1. Access token has expired (they last ~4 hours)")
        print("2. Access token is invalid")
        print("3. App doesn't have required permissions")
        print("\nSolution: Generate a new access token in the Dropbox App Console")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_with_access_token()