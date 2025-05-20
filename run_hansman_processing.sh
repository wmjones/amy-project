#!/bin/bash

# Hansman Processing Script
# Runs the unified processing script for Hansman Syracuse collection

cd /workspaces/amy-project

# Activate virtual environment
source venv/bin/activate

# Run the processing script
python scripts/processing/process_hansman_unified_fixed.py \
    --output-dir /workspaces/amy-project/hansman_full_processing \
    "$@"  # Pass through any additional arguments

echo "Processing completed. Check output in hansman_full_processing directory."