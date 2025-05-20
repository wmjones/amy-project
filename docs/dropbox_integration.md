# Dropbox Integration Guide

This guide explains how to use the Dropbox integration with the File Organizer system.

## Prerequisites

1. A Dropbox account
2. Dropbox app credentials (App Key and App Secret)
3. Python environment with required packages installed

## Setting Up Dropbox App

1. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Click "Create app"
3. Choose:
   - API: Scoped access
   - Access type: Full Dropbox
   - Name: Your app name (e.g., "File Organizer")
4. After creation, note your:
   - App key
   - App secret
5. Set permissions in the "Permissions" tab:
   - files.content.write
   - files.content.read
   - files.metadata.write
   - files.metadata.read
   - account_info.read

## Configuration

Add Dropbox settings to your configuration file or environment variables:

### Configuration File (config.json)

```json
{
  "dropbox": {
    "app_key": "your_app_key",
    "app_secret": "your_app_secret",
    "refresh_token": "your_refresh_token",
    "source_folder": "/file-organizer-uploads",
    "organized_folder": "/Organized",
    "download_batch_size": 10,
    "cleanup_after_organization": false
  }
}
```

### Environment Variables

```bash
export DROPBOX_APP_KEY="your_app_key"
export DROPBOX_APP_SECRET="your_app_secret"
export DROPBOX_REFRESH_TOKEN="your_refresh_token"
```

## Authentication

### First Time Setup

```python
from src.file_access.dropbox_accessor import DropboxAccessor

# First time - will prompt for authorization
accessor = DropboxAccessor(app_key, app_secret)

# Save the refresh token for future use
print(f"Refresh token: {accessor.refresh_token}")
```

### Using Refresh Token

```python
# Subsequent uses - automatic authentication
accessor = DropboxAccessor(app_key, app_secret, refresh_token=saved_token)
```

## Basic Usage

### Listing Files

```python
# List all files
files = accessor.list_files("/")

# List specific file types
pdf_files = accessor.list_files("/Documents", file_types=['.pdf'])

# List files recursively
all_files = accessor.list_files("/", recursive=True)
```

### Downloading Files

```python
# Download single file
local_path = accessor.download_file("/remote/file.pdf", Path("local/file.pdf"))

# Download batch
results = accessor.download_batch(
    ["/file1.pdf", "/file2.jpg"], 
    Path("downloads/")
)
```

### File Operations

```python
# Create folder
accessor.create_folder("/NewFolder")

# Copy file
accessor.copy_file("/source.pdf", "/destination.pdf")

# Move file
accessor.move_file("/old_path.pdf", "/new_path.pdf")

# Delete file
accessor.delete_file("/unwanted.pdf")
```

### Upload Files

```python
# Upload file
metadata = accessor.upload_file(
    Path("local/document.pdf"),
    "/Dropbox/document.pdf"
)

# Upload with overwrite
metadata = accessor.upload_file(
    Path("local/document.pdf"),
    "/Dropbox/document.pdf",
    overwrite=True
)
```

## Organization Integration

### Setting Up Organizer

```python
from src.file_access.dropbox_organizer import DropboxOrganizer

# Initialize components
organizer = DropboxOrganizer(
    config=config,
    dropbox_accessor=accessor,
    metadata_extractor=metadata_extractor,
    organization_engine=organization_engine,
    file_processor=file_processor,
    report_generator=report_generator
)
```

### Preview Organization

```python
# Scan and preview without making changes
preview = organizer.scan_and_preview(
    source_folder="/Uploads",
    file_types=['.pdf', '.jpg', '.docx']
)

print(f"Found {preview['total_files']} files")
print("Sample movements:")
for movement in preview['sample_movements']:
    print(f"{movement['source']} -> {movement['destination']}")
```

### Organize Files

```python
# Organize files with progress tracking
progress_tracker = ProgressTracker()

results = organizer.organize_dropbox_files(
    source_folder="/Uploads",
    progress_tracker=progress_tracker
)

print(f"Processed: {results['processed']}")
print(f"Successful: {results['successful']}")
print(f"Failed: {results['failed']}")
```

### View Organization Stats

```python
# Get statistics about organized files
stats = organizer.get_organization_stats()

print(f"Total organized: {stats['total_organized']}")
print(f"Total size: {stats['total_size']:,} bytes")
print("Files by folder:")
for folder, count in stats['by_folder'].items():
    print(f"  {folder}: {count}")
```

## Advanced Features

### Custom Organization Rules

```python
# Add custom rule for Dropbox organization
custom_rule = {
    'name': 'Dropbox Receipts by Month',
    'priority': 1,
    'conditions': {
        'document_type': 'receipt',
        'dropbox.original_path': {'$starts_with': '/Receipts'}
    },
    'path_template': 'Receipts/{dates.document_date|%Y-%m}/{filename}'
}

config.add_rule(custom_rule)
```

### Batch Processing

```python
# Process files in batches
batch_size = 50

for i in range(0, len(files), batch_size):
    batch = files[i:i + batch_size]
    results = organizer._process_batch(batch, progress_tracker)
```

### Error Handling

```python
from src.utils.error_handler import ErrorHandler

error_handler = ErrorHandler()

try:
    results = organizer.organize_dropbox_files()
except Exception as e:
    result, error = error_handler.handle_error(
        e, 
        "dropbox_organization",
        retry_func=lambda: organizer.organize_dropbox_files()
    )
```

## Performance Considerations

1. **Rate Limits**: Dropbox API has rate limits. The accessor handles basic rate limiting, but for large operations, consider:
   - Using batch operations
   - Implementing delays between requests
   - Monitoring API usage

2. **Large Files**: For files over 150MB, the accessor automatically uses chunked upload/download

3. **Concurrent Operations**: Limit concurrent operations to avoid overwhelming the API:
   ```python
   max_concurrent = 5
   results = accessor.download_batch(files, local_dir, max_concurrent=max_concurrent)
   ```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure app key and secret are correct
   - Check if refresh token has expired
   - Verify app permissions are set correctly

2. **File Not Found**
   - Check the path format (should start with "/")
   - Verify file exists in Dropbox
   - Check for case sensitivity

3. **Permission Errors**
   - Ensure app has required permissions
   - Check if file is in a shared folder with restricted access

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src.file_access.dropbox_accessor')
logger.setLevel(logging.DEBUG)
```

## Security Best Practices

1. **Protect Credentials**
   - Never commit API keys to version control
   - Use environment variables or secure vaults
   - Rotate refresh tokens periodically

2. **Minimize Permissions**
   - Only request necessary Dropbox permissions
   - Use app folders if full Dropbox access isn't needed

3. **Secure Local Storage**
   - Encrypt sensitive data stored locally
   - Clean up temporary files after processing

## Example Workflow

```python
# Complete workflow example
import sys
from pathlib import Path

# 1. Setup
config = ConfigManager()
accessor = DropboxAccessor(
    config.get('dropbox.app_key'),
    config.get('dropbox.app_secret'),
    config.get('dropbox.refresh_token')
)

# 2. Initialize organizer
organizer = DropboxOrganizer(config, accessor, ...)

# 3. Preview organization
preview = organizer.scan_and_preview()
print(f"Will organize {preview['total_files']} files")

# 4. Confirm and organize
if input("Proceed? (y/n): ").lower() == 'y':
    results = organizer.organize_dropbox_files(
        progress_tracker=ProgressTracker()
    )
    
    # 5. Generate report
    report_gen = ReportGenerator()
    report = report_gen.generate_summary_report(
        total_files=results['processed'],
        successful_files=results['successful'],
        failed_files=results['failed']
    )
    print(report)
```

## Integration with Cloud Services

The Dropbox integration can be combined with other cloud services:

```python
# Example: Sync organized files to Google Drive
def sync_to_drive(dropbox_path, drive_service):
    # Download from Dropbox
    temp_file = accessor.download_file(dropbox_path, Path("/tmp/temp"))
    
    # Upload to Google Drive
    drive_service.upload(temp_file, folder_id="...")
    
    # Clean up
    temp_file.unlink()
```

This integration provides seamless file organization capabilities for Dropbox users while maintaining all the powerful features of the File Organizer system.