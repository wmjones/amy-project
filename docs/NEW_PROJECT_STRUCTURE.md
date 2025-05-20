# Project Structure After Reorganization

The project has been reorganized to have a cleaner root directory and better organization of scripts and data. Here's the new structure:

## Directory Structure

```
amy-project/
├── src/                    # Core application code
├── scripts/               # All standalone scripts
│   ├── processing/        # Data processing scripts
│   ├── setup/            # Setup and configuration scripts
│   ├── testing/          # Test and debug scripts
│   └── evaluation/       # Analysis and evaluation scripts
├── workspace/            # All working data and outputs
│   ├── downloads/        # Downloaded files (cached)
│   │   └── hansman/     # Hansman project downloads
│   ├── processed/        # Processed/organized data
│   │   └── hansman/     # Hansman processed output
│   ├── reports/          # Generated reports
│   │   └── hansman/     # Hansman reports
│   ├── cache/           # Temporary cache files
│   └── temp/            # Temporary working files
├── data/                # Static/sample data only
├── config/              # Configuration files
├── docs/                # Documentation
├── examples/            # Demo scripts
├── tests/               # Unit tests
└── tasks/               # Task management files
```

## What Changed

1. **Scripts Organization**: All ad hoc scripts moved to `/scripts/` with subdirectories:
   - `/scripts/processing/`: Contains all `process_hansman_*.py` scripts
   - `/scripts/setup/`: Contains `check_env.py`, `get_dropbox_token.py`, etc.
   - `/scripts/testing/`: Contains `test_*.py` scripts
   - `/scripts/evaluation/`: Contains `ocr_*.py` and evaluation scripts

2. **Data Organization**: All working data moved to `/workspace/`:
   - `/workspace/downloads/hansman/`: All downloaded Hansman files (prevents re-downloads)
   - `/workspace/processed/hansman/`: OCR results, organized files, preprocessing results
   - `/workspace/reports/hansman/`: All generated reports and summaries

3. **New Utilities**:
   - `src/utils/paths.py`: Centralized path constants for consistent file access

## Using the New Structure

### Running Scripts

```bash
# From project root
python scripts/processing/process_hansman_simple.py
python scripts/setup/check_env.py
python scripts/testing/test_modules.py
```

### Accessing Data

```python
from src.utils.paths import *

# Access workspace directories
downloads = HANSMAN_DOWNLOADS  # workspace/downloads/hansman
processed = HANSMAN_PROCESSED  # workspace/processed/hansman
reports = HANSMAN_REPORTS      # workspace/reports/hansman
```

### Benefits

1. **Clean Root**: Only essential files in root directory
2. **No Re-downloads**: Downloads cached in `workspace/downloads/`
3. **Easy Cleanup**: Can safely delete `workspace/temp/` or `workspace/cache/`
4. **Git-friendly**: Working data excluded from version control
5. **Scalable**: Easy to add new projects alongside Hansman

## Migration Notes

- Old directories (`hansman_*`, `preprocessing_results/`, etc.) are still present but will be removed after verification
- All scripts have been updated to use the new paths
- The .gitignore has been updated to exclude workspace data while keeping important reports