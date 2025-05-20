# Setup Guide: Process Hansman Syracuse Photo Docs

This guide will help you set up the file organization project to process all files in the "Hansman Syracuse photo docs July 2015" Dropbox folder and save the results locally.

## Prerequisites

✅ Dropbox access token (already configured)
✅ Anthropic API key (already in .env)
✅ Python environment (already set up)

## Step 1: Create Processing Script

Create a script to process all files from the Dropbox folder:

```python
#!/usr/bin/env python3
"""Process all files in Hansman Syracuse photo docs folder."""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.file_access.dropbox_accessor import DropboxAccessor
from src.utils.config_manager import ConfigManager
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor
from src.organization_logic.engine import OrganizationEngine
from src.utils.report_generator import ReportGenerator
from src.utils.batch_processor import BatchProcessor
from dotenv import load_dotenv

# Load environment
load_dotenv()

def process_hansman_folder():
    """Process all files in the Hansman Syracuse folder."""
    
    # Configuration
    DROPBOX_FOLDER = "/Hansman Syracuse photo docs July 2015"
    LOCAL_OUTPUT = Path("./hansman_processed")
    BATCH_SIZE = 10
    
    # Create output directories
    LOCAL_OUTPUT.mkdir(exist_ok=True)
    (LOCAL_OUTPUT / "downloads").mkdir(exist_ok=True)
    (LOCAL_OUTPUT / "organized").mkdir(exist_ok=True)
    (LOCAL_OUTPUT / "reports").mkdir(exist_ok=True)
    
    # Initialize components
    config = ConfigManager()
    
    # Dropbox accessor
    dropbox_accessor = DropboxAccessor(
        app_key=os.getenv('DROPBOX_APP_KEY'),
        app_secret=os.getenv('DROPBOX_APP_SECRET'),
        access_token=os.getenv('DROPBOX_ACCESS_TOKEN')
    )
    
    # Claude client
    claude_client = ClaudeClient(
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        model=config.get('api.claude_model', 'claude-3-7-sonnet-20250219')
    )
    
    # Metadata extractor
    metadata_extractor = MetadataExtractor(claude_client)
    
    # Organization engine
    org_engine = OrganizationEngine(config)
    
    # Report generator
    report_generator = ReportGenerator()
    
    # Batch processor
    batch_processor = BatchProcessor(
        config=config,
        claude_client=claude_client,
        metadata_extractor=metadata_extractor,
        organization_engine=org_engine
    )
    
    print(f"Starting processing of {DROPBOX_FOLDER}")
    print(f"Output directory: {LOCAL_OUTPUT}")
    print("-" * 50)
    
    try:
        # List all files in the folder
        print("Listing files...")
        files = dropbox_accessor.list_files(DROPBOX_FOLDER, recursive=True)
        print(f"Found {len(files)} files")
        
        # Filter for processable files
        processable_extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.docx', '.txt']
        files_to_process = [f for f in files if f.extension.lower() in processable_extensions]
        print(f"Processing {len(files_to_process)} supported files")
        
        # Download files in batches
        for i in range(0, len(files_to_process), BATCH_SIZE):
            batch = files_to_process[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(len(files_to_process) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Download files
            download_paths = []
            for file in batch:
                try:
                    local_path = LOCAL_OUTPUT / "downloads" / file.name
                    print(f"  Downloading {file.name}...")
                    dropbox_accessor.download_file(file.path, local_path, show_progress=False)
                    download_paths.append(local_path)
                except Exception as e:
                    print(f"  Error downloading {file.name}: {e}")
            
            # Process downloaded files
            if download_paths:
                results = batch_processor.process_files(download_paths)
                
                # Organize files based on metadata
                for result in results:
                    if result.get('success'):
                        metadata = result.get('metadata', {})
                        file_path = result.get('file_path')
                        
                        # Determine organization path
                        org_path = org_engine.get_organization_path(metadata, Path(file_path).name)
                        dest_path = LOCAL_OUTPUT / "organized" / org_path
                        
                        # Create directory and move file
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        Path(file_path).rename(dest_path)
                        print(f"  Organized: {Path(file_path).name} -> {org_path}")
        
        # Generate reports
        print("\nGenerating reports...")
        report_data = {
            'total_files': len(files),
            'processed_files': len(files_to_process),
            'timestamp': datetime.now().isoformat(),
            'source_folder': DROPBOX_FOLDER
        }
        
        # Save reports
        report_generator.generate_summary_report(
            report_data,
            LOCAL_OUTPUT / "reports" / "summary.json"
        )
        
        report_generator.generate_html_report(
            report_data,
            LOCAL_OUTPUT / "reports" / "summary.html"
        )
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {LOCAL_OUTPUT}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    process_hansman_folder()
```

Save this as `process_hansman_folder.py` in the project root.

## Step 2: Create Simplified Runner Script

For easier execution, create a simplified script:

```python
#!/usr/bin/env python3
"""Simple runner for Hansman folder processing."""

import os
from pathlib import Path
import dropbox
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

def process_simple():
    """Simple processing of Hansman files."""
    
    # Configuration
    DROPBOX_FOLDER = "/Hansman Syracuse photo docs July 2015"
    OUTPUT_DIR = Path("./hansman_results")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Connect to Dropbox
    dbx = dropbox.Dropbox(
        app_key=os.getenv('DROPBOX_APP_KEY'),
        app_secret=os.getenv('DROPBOX_APP_SECRET'),
        oauth2_access_token=os.getenv('DROPBOX_ACCESS_TOKEN')
    )
    
    print(f"Processing files from: {DROPBOX_FOLDER}")
    
    # List files
    result = dbx.files_list_folder(DROPBOX_FOLDER)
    files = []
    
    while True:
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                files.append({
                    'name': entry.name,
                    'path': entry.path_display,
                    'size': entry.size,
                    'modified': str(entry.server_modified)
                })
        
        if not result.has_more:
            break
            
        result = dbx.files_list_folder_continue(result.cursor)
    
    print(f"Found {len(files)} files")
    
    # Save file list
    with open(OUTPUT_DIR / 'file_list.json', 'w') as f:
        json.dump(files, f, indent=2)
    
    # Create summary
    summary = {
        'source_folder': DROPBOX_FOLDER,
        'total_files': len(files),
        'processed_date': datetime.now().isoformat(),
        'file_types': {}
    }
    
    # Count file types
    for file in files:
        ext = Path(file['name']).suffix.lower()
        summary['file_types'][ext] = summary['file_types'].get(ext, 0) + 1
    
    # Save summary
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"- File list: {OUTPUT_DIR}/file_list.json")
    print(f"- Summary: {OUTPUT_DIR}/summary.json")

if __name__ == "__main__":
    process_simple()
```

Save this as `process_hansman_simple.py`.

## Step 3: Update Configuration

Update your `.env` file to ensure all necessary credentials are present:

```env
# Required for Dropbox
DROPBOX_APP_KEY=your_app_key
DROPBOX_APP_SECRET=your_app_secret  
DROPBOX_ACCESS_TOKEN=your_access_token

# Required for Claude AI processing
ANTHROPIC_API_KEY=your_anthropic_api_key

# Processing settings
BATCH_SIZE=10
MAX_FILE_SIZE=52428800  # 50MB
```

## Step 4: Run the Processing

### Option A: Simple Processing (List files only)
```bash
cd /workspaces/amy-project
source venv/bin/activate
python process_hansman_simple.py
```

This will:
- List all files in the folder
- Save file information to `./hansman_results/file_list.json`
- Create a summary in `./hansman_results/summary.json`

### Option B: Full Processing (Download and organize)
```bash
cd /workspaces/amy-project
source venv/bin/activate
python process_hansman_folder.py
```

This will:
- Download files from Dropbox
- Extract metadata using Claude AI
- Organize files based on content
- Generate detailed reports

## Step 5: Check Results

Results will be saved to:
- `./hansman_results/` (simple processing)
- `./hansman_processed/` (full processing)
  - `downloads/` - Original downloaded files
  - `organized/` - Files organized by metadata
  - `reports/` - Processing reports

## Folder Structure After Processing

```
amy-project/
├── hansman_results/          # Simple processing output
│   ├── file_list.json       # List of all files
│   └── summary.json         # Processing summary
│
└── hansman_processed/       # Full processing output
    ├── downloads/           # Original files
    ├── organized/           # Organized by content
    │   ├── photos/
    │   ├── documents/
    │   └── ...
    └── reports/            # Processing reports
        ├── summary.json
        └── summary.html
```

## Customization Options

### 1. Filter by file type
```python
# Only process images
files_to_process = [f for f in files if f.extension.lower() in ['.jpg', '.jpeg', '.png']]
```

### 2. Limit number of files
```python
# Process only first 50 files
files_to_process = files_to_process[:50]
```

### 3. Custom organization rules
```python
# Add custom rules to organize files
config.add_rule({
    'name': 'Hansman photos by date',
    'conditions': {'file_type': 'image'},
    'path_template': 'Photos/{date_taken|%Y-%m}/{filename}'
})
```

## Troubleshooting

### Access Token Expired
If you get authentication errors, generate a new access token and update `.env`.

### Memory Issues
Reduce `BATCH_SIZE` in the script to process fewer files at once.

### Network Timeouts
Add retry logic or process files individually.

## Next Steps

1. Run the simple script first to see what files are available
2. Adjust the processing script based on your needs
3. Run the full processing for complete organization
4. Review the generated reports

The project is now configured to process all files from the Hansman Syracuse folder!