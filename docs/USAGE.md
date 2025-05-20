# Usage Guide

This guide provides detailed examples and use cases for the File Organizer with Claude AI.

## Command Line Interface

### Basic Usage

```bash
# Process files from a local directory
python -m src.app --source /path/to/files --target /path/to/organized

# Dry run to preview changes
python -m src.app --source /path/to/files --dry-run

# Use a specific configuration file
python -m src.app --config /path/to/config.json
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to configuration file | `--config config/production.json` |
| `--source` | Source directory or Dropbox path | `--source ~/Documents/unfiled` |
| `--target` | Target directory for organized files | `--target ~/Documents/organized` |
| `--dry-run` | Preview changes without moving files | `--dry-run` |
| `--mode` | File operation mode (copy or move) | `--mode copy` |
| `--verbose` | Enable verbose output | `--verbose` |
| `--log-level` | Set logging level | `--log-level DEBUG` |

## Common Use Cases

### 1. Organizing Personal Documents

Process documents from Downloads folder:

```bash
python -m src.app \
  --source ~/Downloads \
  --target ~/Documents/Organized \
  --mode move
```

Configuration for personal documents:
```json
{
  "organization": {
    "rules": [
      {
        "name": "Personal photos",
        "conditions": {"document_type": "photo"},
        "path_template": "Photos/{dates.taken_date|%Y}/{dates.taken_date|%m-%B}"
      },
      {
        "name": "Tax documents",
        "conditions": {"tags": {"$contains": "tax"}},
        "path_template": "Financial/Taxes/{dates.document_date|%Y}"
      }
    ]
  }
}
```

### 2. Business Document Processing

Organize invoices and contracts:

```bash
python -m src.app \
  --config config/business.json \
  --source /company/inbox \
  --target /company/organized
```

Business configuration example:
```json
{
  "organization": {
    "rules": [
      {
        "name": "Client invoices",
        "conditions": {
          "document_type": "invoice",
          "entities.organizations": {"$exists": true}
        },
        "path_template": "Clients/{entities.organizations[0]}/Invoices/{dates.document_date|%Y}"
      },
      {
        "name": "Contracts by year",
        "conditions": {"document_type": "contract"},
        "path_template": "Legal/Contracts/{dates.document_date|%Y}/{entities.organizations[0]}"
      }
    ]
  }
}
```

### 3. Batch Processing with Progress Tracking

Process large collections with monitoring:

```bash
python -m src.app \
  --source /archive/scanned_documents \
  --target /archive/organized \
  --verbose \
  --config config/batch_processing.json
```

Batch processing configuration:
```json
{
  "processing": {
    "batch_size": 25,
    "max_workers": 8
  },
  "ui": {
    "show_progress": true,
    "update_interval": 1
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/batch_process.log"
  }
}
```

### 4. Dropbox Integration

Process files from Dropbox:

```bash
python -m src.app \
  --config config/dropbox.json \
  --source dropbox:/unorganized \
  --target dropbox:/organized
```

Dropbox configuration:
```json
{
  "source": {
    "type": "dropbox"
  },
  "dropbox": {
    "app_key": "your_app_key",
    "app_secret": "your_app_secret",
    "source_folder": "/unorganized"
  },
  "organization": {
    "base_directory": "/organized"
  }
}
```

### 5. OCR Processing for Scanned Documents

Process scanned documents with OCR:

```bash
python -m src.app \
  --source /scans \
  --target /documents \
  --config config/ocr_enabled.json
```

OCR configuration:
```json
{
  "processing": {
    "enable_ocr": true,
    "ocr_languages": ["eng", "fra", "deu"]
  }
}
```

## Advanced Usage

### Custom Organization Rules

Create complex organization rules:

```json
{
  "organization": {
    "rules": [
      {
        "name": "High-priority financial",
        "priority": 1,
        "conditions": {
          "document_type": {"$in": ["invoice", "receipt"]},
          "amounts.total": {"$gt": 1000},
          "tags": {"$contains": "urgent"}
        },
        "path_template": "Priority/Financial/{dates.document_date|%Y-%m}"
      },
      {
        "name": "Project documents",
        "priority": 2,
        "conditions": {
          "custom": "metadata.project_id is not None"
        },
        "path_template": "Projects/{metadata.project_id}/{document_type}"
      }
    ]
  }
}
```

### Programmatic Usage

Using the package in Python scripts:

```python
from src.app import FileOrganizerApp
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor

# Initialize the application
app = FileOrganizerApp(config_file="config/custom.json")
app.initialize()

# Run the organization process
success = app.run(
    source_override="/path/to/files",
    target_override="/path/to/organized",
    dry_run_override=False
)
```

### Processing Specific File Types

Process only specific file types:

```bash
python -m src.app \
  --source /documents \
  --target /organized \
  --config config/images_only.json
```

Configuration for specific file types:
```json
{
  "processing": {
    "file_types": [".jpg", ".jpeg", ".png", ".tiff"]
  }
}
```

### Metadata Extraction and Storage

Extract and store metadata for later use:

```python
from src.metadata_extraction.storage import MetadataStorage

# Initialize storage
storage = MetadataStorage("./metadata", use_database=True)

# Search stored metadata
results = storage.search_metadata(
    category="financial",
    date_after="2024-01-01"
)

# Export metadata
storage.export_metadata("metadata_export.json", format="json")
```

## Workflow Examples

### Daily Document Processing

Automated daily processing script:

```bash
#!/bin/bash
# daily_organize.sh

DATE=$(date +%Y%m%d)
LOG_FILE="logs/organize_${DATE}.log"

python -m src.app \
  --source ~/Downloads \
  --target ~/Documents/Organized \
  --mode move \
  --config config/daily.json \
  >> "$LOG_FILE" 2>&1

# Send notification
if [ $? -eq 0 ]; then
    echo "Successfully organized files on ${DATE}" | mail -s "File Organization Complete" user@example.com
else
    echo "File organization failed on ${DATE}" | mail -s "File Organization Error" user@example.com
fi
```

### Monthly Archive Processing

Process monthly archives:

```bash
#!/bin/bash
# monthly_archive.sh

YEAR=$(date +%Y)
MONTH=$(date +%m)
ARCHIVE_PATH="/archives/${YEAR}/${MONTH}"

python -m src.app \
  --source "${ARCHIVE_PATH}/raw" \
  --target "${ARCHIVE_PATH}/organized" \
  --config config/archive.json \
  --verbose
```

### Error Recovery and Resumption

Handle interrupted processing:

```bash
# Start processing with resume capability
python -m src.app \
  --source /large_collection \
  --target /organized \
  --config config/resumable.json \
  --resume-from last_checkpoint.json
```

Configuration for resumable processing:
```json
{
  "processing": {
    "checkpoint_interval": 100,
    "checkpoint_file": "checkpoints/progress.json"
  }
}
```

## Performance Optimization

### Large File Collections

Optimize for large collections:

```json
{
  "processing": {
    "batch_size": 50,
    "max_workers": 12,
    "queue_priority": "size"
  },
  "api": {
    "rate_limit": 30,
    "max_tokens": 1500
  }
}
```

### Memory-Efficient Processing

For systems with limited memory:

```json
{
  "processing": {
    "batch_size": 5,
    "max_workers": 2,
    "max_file_size": 52428800
  }
}
```

## Monitoring and Reporting

### Generate Processing Reports

```bash
python -m src.app \
  --source /documents \
  --target /organized \
  --report-path reports/process_report.html \
  --report-format html
```

### Real-time Monitoring

Monitor processing in real-time:

```bash
# Terminal 1: Run the processor
python -m src.app --source /files --target /organized --verbose

# Terminal 2: Monitor logs
tail -f logs/file_organizer.log | grep -E "(ERROR|SUCCESS)"
```

## Best Practices

1. **Always test with dry-run first**
   ```bash
   python -m src.app --source /important --dry-run
   ```

2. **Use appropriate batch sizes** based on your system
   - Small batches (5-10) for limited resources
   - Large batches (25-50) for powerful systems

3. **Configure rate limits** to avoid API throttling
   - Start with conservative limits
   - Increase gradually based on usage

4. **Regular backups** before processing
   ```bash
   rsync -av /source/ /backup/ && python -m src.app --source /source
   ```

5. **Monitor API usage** to control costs
   - Set up usage alerts
   - Use token limits appropriately

6. **Implement logging** for audit trails
   - Set appropriate log levels
   - Rotate logs regularly

## Troubleshooting Common Issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for solutions to common problems.

## Next Steps

- Review [API_REFERENCE.md](API_REFERENCE.md) for programmatic usage
- Check [DEVELOPER.md](DEVELOPER.md) for extending functionality
- See [examples/](../examples/) directory for more code samples
