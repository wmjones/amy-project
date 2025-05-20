#!/usr/bin/env python3
"""Test basic Dropbox access without search API."""

import os
import dropbox
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_access():
    """Test basic Dropbox access and list files."""
    print("=== Testing Basic Dropbox Access ===\n")
    
    # Get access token
    access_token = os.getenv('DROPBOX_ACCESS_TOKEN')
    
    if not access_token:
        print("Please add DROPBOX_ACCESS_TOKEN to your .env file")
        return
    
    try:
        # Create Dropbox client with access token
        print("Connecting to Dropbox...")
        dbx = dropbox.Dropbox(access_token)
        
        # Test the connection
        account = dbx.users_get_current_account()
        print(f"✓ Connected to account: {account.email}\n")
        
        # Try to list root folder
        print("Listing files in root folder...")
        try:
            result = dbx.files_list_folder("")
            print(f"Found {len(result.entries)} items in root folder:")
            
            for entry in result.entries[:10]:  # Show first 10
                if isinstance(entry, dropbox.files.FolderMetadata):
                    print(f"  📁 {entry.name}/")
                else:
                    print(f"  📄 {entry.name}")
            
            if len(result.entries) > 10:
                print(f"  ... and {len(result.entries) - 10} more items")
                
        except Exception as e:
            print(f"Error listing root folder: {e}")
        
        # Try to find the shared folder
        print("\n\nLooking for shared folder 'Hansman Syracuse photo docs July 2015'...")
        
        # Try different approaches
        approaches = [
            ("Direct path", "/Hansman Syracuse photo docs July 2015"),
            ("Without leading slash", "Hansman Syracuse photo docs July 2015"),
            ("In Shared folder", "/Shared/Hansman Syracuse photo docs July 2015"),
        ]
        
        for approach_name, path in approaches:
            print(f"\n{approach_name}: {path}")
            try:
                result = dbx.files_list_folder(path)
                print(f"✓ Success! Found {len(result.entries)} items")
                
                # Look for our file
                for entry in result.entries:
                    if "100_4247" in entry.name:
                        print(f"\n✓ FOUND THE FILE: {entry.name}")
                        print(f"  Size: {entry.size / (1024*1024):.2f} MB")
                        print(f"  Path: {entry.path_display}")
                        print(f"  ID: {entry.id}")
                        return
                    
                # Show some files to help debug
                print("Files in this folder:")
                for entry in result.entries[:5]:
                    print(f"  - {entry.name}")
                    
            except dropbox.exceptions.ApiError as e:
                error = e.error
                if hasattr(error, 'is_path') and error.is_path():
                    path_error = error.get_path()
                    if hasattr(path_error, 'is_not_found') and path_error.is_not_found():
                        print(f"✗ Folder not found")
                    else:
                        print(f"✗ Path error: {path_error}")
                else:
                    print(f"✗ API error: {e}")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Try to get shared folder metadata
        print("\n\nTrying to get shared folder metadata...")
        try:
            # This requires sharing.read permission
            shared_folders = dbx.sharing_list_folders()
            print(f"Found {len(shared_folders.entries)} shared folders:")
            
            for folder in shared_folders.entries:
                print(f"\n  Folder: {folder.name}")
                print(f"  Access: {folder.access_type}")
                
                if "Hansman Syracuse" in folder.name:
                    print("  ✓ This might be our folder!")
                    
                    # Try to list contents
                    try:
                        ns_path = folder.path_lower
                        print(f"  Trying namespace path: {ns_path}")
                        # You might need to mount the folder first
                    except Exception as e:
                        print(f"  Error accessing: {e}")
                        
        except Exception as e:
            print(f"Error listing shared folders: {e}")
            
    except dropbox.exceptions.AuthError as e:
        print(f"\n✗ Authentication error: {e}")
        print("\nMake sure your access token has these permissions:")
        print("- files.metadata.read")
        print("- files.content.read")
        print("- sharing.read")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_basic_access()