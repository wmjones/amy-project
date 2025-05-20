"""Path constants for standardized project structure."""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"

# Workspace subdirectories
DOWNLOADS_DIR = WORKSPACE / "downloads"
PROCESSED_DIR = WORKSPACE / "processed"
REPORTS_DIR = WORKSPACE / "reports"
CACHE_DIR = WORKSPACE / "cache"
TEMP_DIR = WORKSPACE / "temp"

# Project-specific directories
HANSMAN_DOWNLOADS = DOWNLOADS_DIR / "hansman"
HANSMAN_PROCESSED = PROCESSED_DIR / "hansman"
HANSMAN_REPORTS = REPORTS_DIR / "hansman"

# Script directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROCESSING_SCRIPTS = SCRIPTS_DIR / "processing"
SETUP_SCRIPTS = SCRIPTS_DIR / "setup"
TESTING_SCRIPTS = SCRIPTS_DIR / "testing"
EVALUATION_SCRIPTS = SCRIPTS_DIR / "evaluation"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DOWNLOADS_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        CACHE_DIR,
        TEMP_DIR,
        HANSMAN_DOWNLOADS,
        HANSMAN_PROCESSED,
        HANSMAN_REPORTS,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Call this function when the module is imported
ensure_directories()