#!/usr/bin/env python3
"""Test if all required modules are available."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test all required imports."""
    modules_to_test = [
        ("Dropbox SDK", "import dropbox"),
        ("Dotenv", "from dotenv import load_dotenv"),
        (
            "Dropbox Accessor",
            "from src.file_access.dropbox_accessor import DropboxAccessor",
        ),
        ("Config Manager", "from src.utils.config_manager import ConfigManager"),
        ("Claude Client", "from src.claude_integration.client import ClaudeClient"),
        (
            "Metadata Extractor",
            "from src.metadata_extraction.extractor import MetadataExtractor",
        ),
        (
            "Organization Engine",
            "from src.organization_logic.engine import OrganizationEngine",
        ),
        ("Report Generator", "from src.utils.report_generator import ReportGenerator"),
    ]

    print("Testing module imports...")
    print("-" * 40)

    for name, import_statement in modules_to_test:
        try:
            exec(import_statement)
            print(f"✓ {name}: OK")
        except ImportError as e:
            print(f"✗ {name}: FAILED - {e}")
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")

    print("-" * 40)
    print("Import test complete")


if __name__ == "__main__":
    test_imports()
