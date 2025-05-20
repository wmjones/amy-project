"""
Demo script showing Dropbox integration capabilities.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.file_access.dropbox_accessor import DropboxAccessor
from src.file_access.dropbox_organizer import DropboxOrganizer
from src.metadata.claude_client import ClaudeClient
from src.metadata.extractor import MetadataExtractor
from src.metadata.metadata_storage import MetadataStorage
from src.organization.organization_engine import OrganizationEngine
from src.utils.file_processor import FileProcessor
from src.utils.report_generator import ReportGenerator
from src.utils.progress_tracker import ProgressTracker
from src.utils.config_manager import ConfigManager
from src.utils.rule_engine import RuleEngine


def demonstrate_dropbox_authentication():
    """Demonstrate Dropbox authentication process."""
    print("=== Dropbox Authentication Demo ===\n")

    # Load configuration
    config = ConfigManager()

    # Get Dropbox credentials from config or environment
    app_key = config.get("dropbox.app_key")
    app_secret = config.get("dropbox.app_secret")
    refresh_token = config.get("dropbox.refresh_token")

    if not app_key or not app_secret:
        print("Error: Dropbox app credentials not found in configuration.")
        print("Please set 'dropbox.app_key' and 'dropbox.app_secret' in your config.")
        return None

    print(f"App Key: {app_key[:8]}...")
    print(f"App Secret: {app_secret[:8]}...")

    try:
        if refresh_token:
            print("Using existing refresh token for authentication...")
            accessor = DropboxAccessor(app_key, app_secret, refresh_token=refresh_token)
            print("Successfully authenticated with Dropbox!")
        else:
            print("\nNo refresh token found. Starting OAuth flow...")
            print("This will open a browser for authorization.")
            accessor = DropboxAccessor(app_key, app_secret)
            print("\nAuthentication successful!")
            print("To avoid re-authentication, save the refresh token to your config.")

        return accessor

    except Exception as e:
        print(f"Authentication failed: {e}")
        return None


def demonstrate_file_listing(accessor: DropboxAccessor):
    """Demonstrate listing files in Dropbox."""
    print("\n=== File Listing Demo ===\n")

    try:
        # List files in root
        print("Listing files in Dropbox root...")
        files = accessor.list_files("/", recursive=False)

        if not files:
            print("No files found in root directory.")
        else:
            print(f"Found {len(files)} files:")
            for file in files[:5]:  # Show first 5
                print(f"  - {file.name} ({file.size:,} bytes)")

            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")

        # List specific file types
        print("\nListing PDF and image files...")
        filtered_files = accessor.list_files("/", file_types=[".pdf", ".jpg", ".png"])
        print(f"Found {len(filtered_files)} PDF/image files")

        # Show space usage
        print("\nDropbox space usage:")
        usage = accessor.get_space_usage()
        used_gb = usage["used"] / (1024**3)
        allocated_gb = usage["allocated"] / (1024**3)
        percentage = (usage["used"] / usage["allocated"]) * 100
        print(f"  Used: {used_gb:.2f} GB / {allocated_gb:.2f} GB ({percentage:.1f}%)")

    except Exception as e:
        print(f"Error listing files: {e}")


def demonstrate_file_operations(accessor: DropboxAccessor):
    """Demonstrate basic file operations."""
    print("\n=== File Operations Demo ===\n")

    # Create a test folder
    test_folder = "/FileOrganizerTest"
    print(f"Creating test folder: {test_folder}")

    try:
        accessor.create_folder(test_folder)
        print("Test folder created successfully")
    except Exception as e:
        print(f"Error creating folder: {e}")
        return

    # Create a test file locally
    test_content = f"Test file created at {datetime.now()}"
    local_test_file = Path("test_upload.txt")
    local_test_file.write_text(test_content)

    # Upload test file
    print(f"\nUploading test file to {test_folder}/test_upload.txt")
    try:
        accessor.upload_file(local_test_file, f"{test_folder}/test_upload.txt")
        print("File uploaded successfully")
    except Exception as e:
        print(f"Error uploading file: {e}")

    # Copy the file
    print(f"\nCopying file to {test_folder}/test_copy.txt")
    try:
        accessor.copy_file(
            f"{test_folder}/test_upload.txt", f"{test_folder}/test_copy.txt"
        )
        print("File copied successfully")
    except Exception as e:
        print(f"Error copying file: {e}")

    # Move the copy
    print(f"\nMoving copy to {test_folder}/subfolder/test_moved.txt")
    try:
        accessor.create_folder(f"{test_folder}/subfolder")
        accessor.move_file(
            f"{test_folder}/test_copy.txt", f"{test_folder}/subfolder/test_moved.txt"
        )
        print("File moved successfully")
    except Exception as e:
        print(f"Error moving file: {e}")

    # List files in test folder
    print(f"\nListing files in {test_folder}...")
    try:
        files = accessor.list_files(test_folder, recursive=True)
        for file in files:
            print(f"  - {file.path}")
    except Exception as e:
        print(f"Error listing files: {e}")

    # Clean up
    print("\nCleaning up test files...")
    try:
        # Download file first as a demo
        download_path = Path("downloaded_test.txt")
        accessor.download_file(f"{test_folder}/test_upload.txt", download_path)
        print(f"Downloaded file content: {download_path.read_text()}")
        download_path.unlink()

        # Delete test files
        for file in files:
            accessor.delete_file(file.path)
        print("Test files deleted")

        # Delete test folder
        accessor.delete_file(test_folder)
        print("Test folder deleted")

    except Exception as e:
        print(f"Error during cleanup: {e}")

    # Clean up local file
    if local_test_file.exists():
        local_test_file.unlink()


def demonstrate_organization(accessor: DropboxAccessor):
    """Demonstrate file organization in Dropbox."""
    print("\n=== Dropbox Organization Demo ===\n")

    # Initialize configuration
    config = ConfigManager()

    # Initialize required components
    claude_client = ClaudeClient(config.get("api.anthropic_api_key"))
    metadata_storage = MetadataStorage(":memory:")  # Use in-memory DB for demo
    metadata_extractor = MetadataExtractor(claude_client, metadata_storage)

    rule_engine = RuleEngine()
    organization_engine = OrganizationEngine(config, rule_engine)

    file_processor = FileProcessor()
    report_generator = ReportGenerator()
    progress_tracker = ProgressTracker()

    # Create organizer
    organizer = DropboxOrganizer(
        config=config,
        dropbox_accessor=accessor,
        metadata_extractor=metadata_extractor,
        organization_engine=organization_engine,
        file_processor=file_processor,
        report_generator=report_generator,
    )

    # Preview organization
    print("Scanning Dropbox for files to organize...")
    try:
        preview = organizer.scan_and_preview()

        print(f"\nFound {preview['total_files']} files to organize")
        print("\nFile type distribution:")
        for file_type, count in preview["by_type"].items():
            print(f"  {file_type}: {count}")

        print("\nFile size distribution:")
        for size_range, count in preview["by_size"].items():
            print(f"  {size_range}: {count}")

        print("\nSample organization movements:")
        for movement in preview["sample_movements"]:
            print(f"  {movement['source']} -> {movement['destination']}")
            print(f"    Rule: {movement['rule']}")

        # Get current organization stats
        stats = organizer.get_organization_stats()
        print(f"\nCurrent organization stats:")
        print(f"  Total organized files: {stats.get('total_organized', 0)}")
        print(f"  Total size: {stats.get('total_size', 0):,} bytes")

    except Exception as e:
        print(f"Error during preview: {e}")


def main():
    """Main demo function."""
    print("Dropbox Integration Demo\n")

    # Step 1: Authentication
    accessor = demonstrate_dropbox_authentication()
    if not accessor:
        print("Authentication failed. Exiting.")
        return

    # Step 2: File listing
    demonstrate_file_listing(accessor)

    # Step 3: File operations
    response = input("\nPerform file operation demos? (y/n): ")
    if response.lower() == "y":
        demonstrate_file_operations(accessor)

    # Step 4: Organization demo
    response = input("\nPerform organization preview? (y/n): ")
    if response.lower() == "y":
        demonstrate_organization(accessor)

    print("\n=== Demo completed! ===")


if __name__ == "__main__":
    main()
