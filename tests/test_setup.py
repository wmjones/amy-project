import os
import sys
from pathlib import Path


def test_project_structure():
    """Test that the project structure is correctly set up."""
    expected_dirs = [
        "src/file_access",
        "src/claude_integration",
        "src/metadata_extraction",
        "src/organization_logic",
        "src/utils",
        "tests",
        "config",
        "logs",
    ]

    for dir_path in expected_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"


def test_dependencies():
    """Test that key dependencies can be imported."""
    try:
        import anthropic
        import PIL
        import PyPDF2
        import docx
        import dotenv
    except ImportError as e:
        assert False, f"Failed to import dependency: {e}"


def test_environment_template():
    """Test that .env.example exists."""
    assert os.path.exists(".env.example"), ".env.example file not found"
