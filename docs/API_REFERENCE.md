# API Reference

This document provides a comprehensive API reference for developers using the File Organizer package.

## Table of Contents

- [Core Classes](#core-classes)
- [Claude Integration](#claude-integration)
- [Metadata Extraction](#metadata-extraction)
- [Organization Logic](#organization-logic)
- [File Access](#file-access)
- [Utilities](#utilities)
- [Examples](#examples)

## Core Classes

### FileOrganizerApp

Main application controller that orchestrates the file organization process.

```python
from src.app import FileOrganizerApp
```

#### Constructor

```python
app = FileOrganizerApp(config_file: Optional[str] = None)
```

**Parameters:**
- `config_file` (str, optional): Path to configuration file

#### Methods

##### initialize()
```python
app.initialize() -> None
```
Initializes all components and loads configuration.

##### run()
```python
app.run(
    source_override: Optional[str] = None,
    target_override: Optional[str] = None,
    dry_run_override: Optional[bool] = None
) -> bool
```

**Parameters:**
- `source_override`: Override source directory
- `target_override`: Override target directory  
- `dry_run_override`: Override dry-run mode

**Returns:** `bool` - Success status

## Claude Integration

### ClaudeClient

Client for interacting with Claude API.

```python
from src.claude_integration.client import ClaudeClient
```

#### Constructor

```python
client = ClaudeClient(
    api_key: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 2000,
    retry_attempts: int = 3,
    retry_delay: float = 1.0
)
```

#### Methods

##### analyze_document()
```python
result = client.analyze_document(
    content: str,
    file_name: str,
    file_type: str,
    custom_prompt: Optional[str] = None
) -> AnalysisResult
```

**Parameters:**
- `content`: Document content
- `file_name`: Name of the file
- `file_type`: File extension
- `custom_prompt`: Optional custom prompt

**Returns:** `AnalysisResult` object

##### analyze_batch()
```python
results = client.analyze_batch(
    documents: List[Dict[str, Any]],
    batch_size: int = 10,
    progress_callback: Optional[Callable] = None
) -> List[AnalysisResult]
```

**Parameters:**
- `documents`: List of document dictionaries
- `batch_size`: Number of documents per batch
- `progress_callback`: Progress callback function

### AnalysisResult

Result from Claude analysis.

```python
@dataclass
class AnalysisResult:
    content: str
    metadata: Dict[str, Any]
    confidence_score: float
    tokens_used: int
    model: str
```

## Metadata Extraction

### MetadataExtractor

Extracts structured metadata from documents.

```python
from src.metadata_extraction.extractor import MetadataExtractor
```

#### Constructor

```python
extractor = MetadataExtractor(claude_client: ClaudeClient)
```

#### Methods

##### extract_metadata()
```python
metadata = extractor.extract_metadata(
    file_content: str,
    file_path: str,
    file_type: str,
    file_size: Optional[int] = None,
    custom_prompt: Optional[str] = None
) -> DocumentMetadata
```

**Parameters:**
- `file_content`: Content of the file
- `file_path`: Path to the file
- `file_type`: Type of file
- `file_size`: Size in bytes
- `custom_prompt`: Custom extraction prompt

### DocumentMetadata

Structured metadata for documents.

```python
@dataclass
class DocumentMetadata:
    document_type: str
    categories: List[str]
    dates: DateInfo
    entities: List[Entity]
    topics: List[str]
    tags: List[str]
    summary: str
    suggested_folder: str
    confidence_score: float
    source_file: str
    processing_timestamp: str
    file_size: Optional[int] = None
    language: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
```

### DateInfo

Date information extracted from documents.

```python
@dataclass
class DateInfo:
    document_date: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    mentioned_dates: List[str] = field(default_factory=list)
```

### Entity

Named entities found in documents.

```python
@dataclass
class Entity:
    name: str
    type: str  # person, organization, location, etc.
    confidence: float = 1.0
```

## Organization Logic

### OrganizationEngine

Determines file organization based on rules.

```python
from src.organization_logic.engine import OrganizationEngine
```

#### Constructor

```python
engine = OrganizationEngine(
    rules_path: Optional[str] = None,
    use_default_rules: bool = True,
    default_folder: str = "Unsorted"
)
```

#### Methods

##### determine_target_location()
```python
path, rule_name = engine.determine_target_location(
    metadata: DocumentMetadata
) -> Tuple[str, str]
```

**Parameters:**
- `metadata`: Document metadata

**Returns:** Tuple of (target_path, rule_name)

##### add_rule()
```python
engine.add_rule(rule_dict: Dict[str, Any]) -> None
```

Adds a custom organization rule.

### OrganizationRule

Defines organization rules.

```python
@dataclass
class OrganizationRule:
    name: str
    conditions: Dict[str, Any]
    path_template: str
    priority: int = 50
    enabled: bool = True
    description: str = ""
```

## File Access

### FileSystemAccessor

Access files from local filesystem.

```python
from src.file_access.local_accessor import FileSystemAccessor
```

#### Constructor

```python
accessor = FileSystemAccessor(base_path: str)
```

#### Methods

##### get_supported_files()
```python
files = accessor.get_supported_files() -> List[FileInfo]
```

Returns list of supported files in the directory.

### DropboxAccessor

Access files from Dropbox.

```python
from src.file_access.dropbox_accessor import DropboxAccessor
```

#### Constructor

```python
accessor = DropboxAccessor(
    app_key: str,
    app_secret: str,
    refresh_token: Optional[str] = None,
    access_token: Optional[str] = None
)
```

#### Methods

##### list_files()
```python
files = accessor.list_files(
    folder_path: str,
    recursive: bool = True,
    file_types: Optional[List[str]] = None
) -> List[DropboxFileInfo]
```

### FileProcessor

Process files for metadata extraction.

```python
from src.file_access.processor import FileProcessor
```

#### Constructor

```python
processor = FileProcessor(enable_ocr: bool = True)
```

#### Methods

##### process_file()
```python
content = processor.process_file(
    file_path: str,
    file_type: Optional[str] = None
) -> str
```

Extracts text content from files.

## Utilities

### ConfigManager

Manages application configuration.

```python
from src.utils.config_manager import ConfigManager
```

#### Constructor

```python
config_manager = ConfigManager(config_file: Optional[Path] = None)
```

#### Properties

- `config`: Dictionary containing configuration

### BatchProcessor

Processes files in batches.

```python
from src.utils.simple_batch_processor import SimpleBatchProcessor
```

#### Constructor

```python
processor = SimpleBatchProcessor(
    claude_client: ClaudeClient,
    file_processor: FileProcessor,
    metadata_extractor: MetadataExtractor,
    rate_limit: int = 10
)
```

#### Methods

##### process_batch()
```python
results = processor.process_batch(
    file_paths: List[str],
    progress_callback: Optional[Callable] = None
) -> List[Tuple[str, Dict, Optional[Exception]]]
```

### ProgressTracker

Tracks processing progress.

```python
from src.utils.progress import ProgressTracker
```

#### Constructor

```python
tracker = ProgressTracker(total_items: int)
```

#### Methods

##### update_progress()
```python
tracker.update_progress(
    item_id: str,
    status: str,
    details: Any = None
) -> None
```

### ReportGenerator

Generates processing reports.

```python
from src.utils.report_generator import ReportGenerator
```

#### Constructor

```python
generator = ReportGenerator(
    progress_tracker: ProgressTracker,
    file_manipulator: FileManipulator
)
```

#### Methods

##### generate_summary_report()
```python
report = generator.generate_summary_report() -> str
```

## Examples

### Basic Usage

```python
from src.app import FileOrganizerApp

# Create and run the application
app = FileOrganizerApp(config_file="config/custom.json")
app.initialize()
success = app.run(source_override="/path/to/files")
```

### Custom Metadata Extraction

```python
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor

# Initialize components
client = ClaudeClient(api_key="your-key")
extractor = MetadataExtractor(client)

# Extract metadata
metadata = extractor.extract_metadata(
    file_content="Document content...",
    file_path="/docs/document.pdf",
    file_type="pdf"
)

print(f"Document type: {metadata.document_type}")
print(f"Categories: {metadata.categories}")
```

### Custom Organization Rules

```python
from src.organization_logic.engine import OrganizationEngine

# Create engine with custom rules
engine = OrganizationEngine(use_default_rules=False)

# Add custom rule
engine.add_rule({
    "name": "Project files",
    "conditions": {
        "tags": {"$contains": "project"}
    },
    "path_template": "Projects/{tags[0]}/{document_type}",
    "priority": 90
})

# Determine location
path, rule = engine.determine_target_location(metadata)
```

### Batch Processing

```python
from src.utils.simple_batch_processor import SimpleBatchProcessor
from src.claude_integration.client import ClaudeClient
from src.file_access.processor import FileProcessor
from src.metadata_extraction.extractor import MetadataExtractor

# Initialize components
client = ClaudeClient()
file_processor = FileProcessor()
extractor = MetadataExtractor(client)

# Create batch processor
batch_processor = SimpleBatchProcessor(
    claude_client=client,
    file_processor=file_processor,
    metadata_extractor=extractor,
    rate_limit=15
)

# Process files
file_paths = ["/docs/file1.pdf", "/docs/file2.jpg"]
results = batch_processor.process_batch(file_paths)

for path, result, error in results:
    if error:
        print(f"Error processing {path}: {error}")
    else:
        print(f"Processed {path}: {result['metadata'].document_type}")
```

### Error Handling

```python
from src.utils.error_handler import ErrorHandler

error_handler = ErrorHandler(max_retries=3)

@error_handler.with_retry
def process_file(file_path):
    # Processing logic here
    pass
```

## Type Hints

All methods use proper type hints for better IDE support:

```python
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
```

## Thread Safety

Most components are thread-safe. For concurrent operations:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_file, path) for path in file_paths]
```

## Extension Points

To extend functionality:

1. Create custom metadata extractors
2. Implement new file accessors
3. Add organization rules
4. Create custom processors

See [DEVELOPER.md](DEVELOPER.md) for detailed extension guide.