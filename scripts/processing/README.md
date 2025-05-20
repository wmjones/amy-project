# Hansman Processing Scripts

This directory contains scripts for processing the Hansman Syracuse photo collection from Dropbox.

## Unified Processing Script

The main script is `process_hansman_unified.py` which consolidates all processing functionality into a single script with smart step checking and force options.

### Features

1. **Smart Step Checking**: The script checks if each processing step has already been completed and skips it to avoid redundant work.
2. **Force Options**: You can force re-running of specific steps (download, OCR, AI, organize) even if they've been completed.
3. **State Persistence**: Processing state is saved to disk, allowing you to resume processing after interruptions.
4. **Comprehensive Reporting**: Generates detailed reports of processing results and errors.

### Usage

```bash
# Basic usage - process all files with smart skipping
python process_hansman_unified.py

# Limit number of files to process (useful for testing)
python process_hansman_unified.py --limit 10

# Force re-download of files
python process_hansman_unified.py --force-download

# Force re-run OCR on all files
python process_hansman_unified.py --force-ocr

# Force re-run AI summarization
python process_hansman_unified.py --force-ai

# Force re-organize files
python process_hansman_unified.py --force-organize

# Force re-run everything
python process_hansman_unified.py --force-all

# Use custom output directory
python process_hansman_unified.py --output-dir ./my_output

# Combine options
python process_hansman_unified.py --limit 20 --force-ocr --output-dir ./test_run
```

### Processing Steps

1. **Download**: Downloads files from Dropbox to local storage
   - Skips files that have already been downloaded (unless --force-download)
   - Tracks download metadata and timestamps

2. **OCR**: Processes images to extract text
   - Uses Tesseract OCR with preprocessing
   - Caches results to avoid reprocessing
   - Saves both text and JSON results

3. **AI Summary**: Analyzes OCR text with Claude AI
   - Generates summaries and extracts metadata
   - Identifies document types, dates, entities
   - Provides archival suggestions

4. **Organization**: Organizes files into logical folders
   - Uses custom rules for Hansman collection
   - Categories include Syracuse Lists, Sequential Photos, Correspondence, etc.
   - Preserves original files while creating organized copies

### Output Structure

```
hansman_results/
├── downloads/          # Original downloaded files
├── ocr_results/        # OCR text and JSON results
├── summaries/          # AI-generated summaries and metadata
├── organized/          # Files organized by category
├── reports/            # Processing reports and summaries
└── .state/            # Processing state and cache
    ├── processing_state.json
    └── ocr_cache/
```

### State File

The processing state is saved in `.state/processing_state.json` and includes:
- Downloaded files and their metadata
- OCR processing status
- AI processing status
- Organization status
- Error tracking

This allows the script to:
- Resume processing after interruptions
- Skip already completed steps
- Track processing history

### Script History

The unified script consolidates functionality from several previous scripts that have been removed:

- `process_hansman_simple.py` - Basic file listing and summary
- `process_hansman_advanced.py` - Included AI analysis
- `process_hansman_full_run.py` - Complete processing pipeline
- `process_hansman_practical.py` - Simplified organization
- `continue_hansman_processing.py` - Resume interrupted processing
- `run_hansman_full_pipeline.py` - Alternative full pipeline

All functionality from these scripts has been integrated into `process_hansman_unified.py` with improvements including smart step checking, state persistence, and force options.

### Environment Variables

Required environment variables (set in `.env` file):
- `DROPBOX_APP_KEY`
- `DROPBOX_APP_SECRET`
- `DROPBOX_ACCESS_TOKEN`
- `ANTHROPIC_API_KEY`

### Error Handling

The script includes comprehensive error handling:
- Continues processing other files if one fails
- Tracks all errors in the state file
- Generates error reports
- Allows retry of failed files with force options

### Performance Notes

- OCR processing can be slow for large images
- AI processing is rate-limited by the Claude API
- The script uses progress bars to show processing status
- Caching significantly speeds up re-runs

### Tips

1. Start with a small `--limit` to test the pipeline
2. Use `--force-ocr` if you've improved OCR settings
3. Use `--force-ai` if you've updated AI prompts
4. Check reports for processing statistics and errors
5. The state file can be edited manually if needed
