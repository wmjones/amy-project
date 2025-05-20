# Project Structure Recommendation Guide

## Current Issues

1. **Root Directory Clutter**: Multiple ad hoc scripts and test files in the root
2. **Data Output Redundancy**: Multiple `hansman_*` directories storing downloaded and processed data
3. **Mixed Purpose Files**: Temporary, test, and production files intermixed
4. **Inconsistent Script Placement**: Processing scripts scattered across root and subdirectories

## Recommended Project Structure

```
amy-project/
├── README.md
├── setup.py
├── requirements.txt
├── pytest.ini
├── package.json
├── .gitignore
├── .env
├── .pre-commit-config.yaml
├── .mcp.json
├── .taskmasterconfig
├── .devcontainer/
├── .claude/
├── config/               # All configuration files
│   ├── business.json
│   ├── personal.json
│   ├── dropbox.json
│   ├── ocr_heavy.json
│   └── ...
├── src/                  # Core application code
│   ├── __init__.py
│   ├── app.py           # Main application entry point
│   ├── claude_integration/
│   ├── file_access/
│   ├── metadata_extraction/
│   ├── organization_logic/
│   ├── integration/
│   ├── optimization/
│   └── utils/
├── scripts/              # All standalone scripts
│   ├── processing/       # Data processing scripts
│   │   ├── process_hansman_simple.py
│   │   ├── process_hansman_advanced.py
│   │   ├── process_hansman_full_run.py
│   │   ├── continue_hansman_processing.py
│   │   └── run_hansman_full_pipeline.py
│   ├── setup/           # Setup and configuration scripts
│   │   ├── check_env.py
│   │   ├── get_dropbox_token.py
│   │   ├── diagnose_dropbox.py
│   │   └── simple_dropbox_test.py
│   ├── testing/         # Test and debug scripts
│   │   ├── test_app_cli.py
│   │   ├── test_basic_access.py
│   │   ├── test_dropbox_connection.py
│   │   ├── test_shared_folder.py
│   │   └── test_with_access_token.py
│   └── evaluation/      # Analysis and evaluation scripts
│       ├── ocr_evaluation.py
│       ├── ocr_preprocessing.py
│       └── ocr_processor.py
├── data/                # Static/sample data only
│   └── hansman_samples/
├── workspace/           # All working data and outputs
│   ├── downloads/       # Downloaded files from Dropbox
│   │   └── hansman/    # Hansman project downloads
│   ├── processed/       # Processed/organized data
│   │   └── hansman/    # Hansman processed output
│   ├── reports/         # Generated reports
│   │   └── hansman/
│   ├── cache/          # Temporary cache files
│   └── temp/           # Temporary working files
├── tests/              # Unit tests
│   ├── __init__.py
│   ├── test_*.py
│   └── ...
├── docs/               # Documentation
│   ├── *.md
│   └── diagrams/
├── examples/           # Demo scripts and examples
├── logs/              # Application logs
└── tasks/             # Task management files
```

## Implementation Guide

### 1. Create New Directory Structure

```bash
# Create workspace directories
mkdir -p workspace/{downloads,processed,reports,cache,temp}
mkdir -p workspace/downloads/hansman
mkdir -p workspace/processed/hansman
mkdir -p workspace/reports/hansman

# Create scripts subdirectories
mkdir -p scripts/{processing,setup,testing,evaluation}
```

### 2. Move Files to Appropriate Locations

#### Ad hoc Scripts → scripts/
```bash
# Processing scripts
mv process_hansman_*.py scripts/processing/
mv run_hansman_full_pipeline.py scripts/processing/
mv continue_hansman_processing.py scripts/processing/

# Setup scripts
mv check_env.py scripts/setup/
mv get_dropbox_token.py scripts/setup/
mv diagnose_dropbox.py scripts/setup/
mv simple_dropbox_test.py scripts/setup/

# Test scripts
mv test_*.py scripts/testing/

# Evaluation scripts
mv ocr_*.py scripts/evaluation/
```

#### Data Files → workspace/
```bash
# Move downloaded data
mv hansman_full_run/downloads/* workspace/downloads/hansman/
mv hansman_organized/downloads/* workspace/downloads/hansman/

# Move processed data
mv hansman_full_run/ocr_results workspace/processed/hansman/
mv hansman_organized/organized workspace/processed/hansman/
mv preprocessing_results workspace/processed/hansman/

# Move reports
mv hansman_full_run/reports/* workspace/reports/hansman/
mv hansman_organized/reports/* workspace/reports/hansman/
mv hansman_results/* workspace/reports/hansman/
mv output/reports/* workspace/reports/hansman/
```

### 3. Update .gitignore

Add the following entries to `.gitignore`:

```gitignore
# Workspace directories
workspace/downloads/
workspace/processed/
workspace/cache/
workspace/temp/

# Keep reports but ignore large files
workspace/reports/**/*.json
!workspace/reports/**/*summary.json
workspace/reports/**/*.html

# Legacy directories (remove after migration)
hansman_*/
preprocessing_results/
ingestion_test/
report_demo_temp/
output/
```

### 4. Update Configuration Files

Update all configuration files to use the new paths:

```json
{
  "output": {
    "downloads": "workspace/downloads/hansman",
    "processed": "workspace/processed/hansman",
    "reports": "workspace/reports/hansman"
  }
}
```

### 5. Create Path Constants Module

Create `src/utils/paths.py`:

```python
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

# Ensure directories exist
for directory in [DOWNLOADS_DIR, PROCESSED_DIR, REPORTS_DIR, CACHE_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
```

### 6. Benefits of This Structure

1. **Clean Root Directory**: Only essential files in root
2. **Logical Organization**: Clear separation of code, scripts, data, and outputs
3. **Prevents Re-downloads**: `workspace/downloads` acts as a persistent cache
4. **Easy Cleanup**: Can delete entire `workspace/temp` or `workspace/cache` safely
5. **Git-friendly**: Working data excluded from version control
6. **Scalable**: Easy to add new projects alongside Hansman

### 7. Migration Checklist

- [ ] Create new directory structure
- [ ] Move scripts to `scripts/` subdirectories
- [ ] Move data to `workspace/` subdirectories
- [ ] Update .gitignore
- [ ] Update configuration files
- [ ] Create path constants module
- [ ] Update import statements in moved scripts
- [ ] Test scripts with new paths
- [ ] Remove old directories after verification
- [ ] Update documentation

### 8. Best Practices Going Forward

1. **Never put data in root**: Use `workspace/` for all data operations
2. **Organize scripts by purpose**: Use `scripts/` subdirectories
3. **Keep downloads**: Store in `workspace/downloads/` to avoid re-downloading
4. **Clean temp files**: Regularly clean `workspace/temp/`
5. **Document outputs**: Always generate reports in `workspace/reports/`
