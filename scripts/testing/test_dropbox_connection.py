#!/usr/bin/env python3
"""Test Dropbox connection by finding a specific file in a shared folder."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.file_access.dropbox_accessor import DropboxAccessor
from src.utils.config_manager import ConfigManager

# Load dotenv
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

def test_find_file():
    """Test finding file 100_4247 in shared folder 'Hansman Syracuse photo docs July 2015'."""
    print("=== Testing Dropbox Connection ===\n")
    
    # Initialize configuration
    config = ConfigManager()
    
    # Check if we have Dropbox credentials
    app_key = config.get('dropbox.app_key') or os.getenv('DROPBOX_APP_KEY')
    app_secret = config.get('dropbox.app_secret') or os.getenv('DROPBOX_APP_SECRET')
    refresh_token = config.get('dropbox.refresh_token') or os.getenv('DROPBOX_REFRESH_TOKEN')
    
    if not all([app_key, app_secret, refresh_token]):
        print("Error: Dropbox credentials not found in config or environment variables.")
        print("Please set DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN")
        return
    
    # Initialize Dropbox accessor
    accessor = DropboxAccessor(
        app_key=app_key,
        app_secret=app_secret,
        refresh_token=refresh_token
    )
    
    # Authentication happens in __init__, so just test the connection
    print("Testing Dropbox connection...")
    try:
        # Test by getting current account info
        account = accessor.client.users_get_current_account()
        print(f"✓ Successfully connected to Dropbox account: {account.email}\n")
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        return
    
    # Search for the file
    print(f"Searching for file '100_4247' in shared folder 'Hansman Syracuse photo docs July 2015'...")
    print("-" * 80)
    
    try:
        # First, try to list files in the shared folder
        shared_folder_path = "/Hansman Syracuse photo docs July 2015"
        
        # Use search_files method to find the specific file
        search_results = accessor.search_files("100_4247")
        
        # Filter results to only those in the specified folder
        found_files = []
        for file in search_results:
            # Check if the file is in the target folder
            if "Hansman Syracuse photo docs July 2015" in file.path:
                found_files.append(file)
                
        if found_files:
            print(f"\n✓ Found {len(found_files)} matching file(s):\n")
            for file in found_files:
                print(f"  Name: {file.name}")
                print(f"  Path: {file.path}")
                print(f"  Size: {file.size / (1024*1024):.2f} MB")
                print(f"  Extension: {file.extension}")
                print(f"  Modified: {file.modified}")
                print("-" * 40)
        else:
            print(f"\n✗ File '100_4247' not found in folder '{shared_folder_path}'")
            
            # Try listing all files in the folder to help debug
            print(f"\nListing all files in '{shared_folder_path}'...")
            try:
                all_files = accessor.list_files(shared_folder_path, recursive=False)
                if all_files:
                    print(f"Found {len(all_files)} files in the folder:")
                    for file in all_files[:10]:  # Show first 10 files
                        print(f"  - {file.name}")
                    if len(all_files) > 10:
                        print(f"  ... and {len(all_files) - 10} more files")
                else:
                    print("No files found in the folder")
            except Exception as e:
                print(f"Error listing folder contents: {e}")
                
    except Exception as e:
        print(f"\n✗ Error searching for file: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # If it's a permissions error, provide helpful information
        if "PathError" in str(type(e)):
            print("\nNote: This might be a shared folder that requires specific permissions.")
            print("Make sure your Dropbox app has access to shared folders.")
        elif "AuthError" in str(type(e)):
            print("\nNote: Authentication issue. Please check your credentials.")
    
    print("\n" + "-" * 80)
    print("Test completed")

if __name__ == "__main__":
    test_find_file()