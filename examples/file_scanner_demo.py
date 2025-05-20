#!/usr/bin/env python3
"""
Demo script showing how to use the FileSystemAccessor
"""

from src.file_access.local_accessor import FileSystemAccessor
import sys
import logging
from pathlib import Path


def main():
    """Main function to demonstrate file scanner."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get directory from command line or use current directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    print(f"Scanning directory: {directory}\n")

    try:
        # Create file accessor
        accessor = FileSystemAccessor(directory)

        # Get directory statistics
        stats = accessor.get_directory_stats()
        print("Directory Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Supported files: {stats['supported_files']}")
        print(f"  Total size: {stats['total_size']:,} bytes")
        print("\nFiles by extension:")
        for ext, count in stats["by_extension"].items():
            print(f"  {ext}: {count}")

        # Get supported files
        print("\n\nSupported files:")
        supported_files = accessor.get_supported_files()
        for file in supported_files[:10]:  # Show first 10
            print(f"  {file.name} ({file.size:,} bytes)")

        if len(supported_files) > 10:
            print(f"  ... and {len(supported_files) - 10} more files")

        # Example: Filter by extension
        print("\n\nImage files:")
        images = accessor.filter_by_extension([".jpg", ".jpeg", ".png"])
        for img in images[:5]:
            print(f"  {img.name} - {img.modified_time}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
