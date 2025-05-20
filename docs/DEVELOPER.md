# Developer Guide

This guide provides information for developers who want to extend or contribute to the File Organizer project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Extension Points](#extension-points)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Contributing](#contributing)
- [Code Style](#code-style)

## Architecture Overview

The File Organizer follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐
│   CLI/App       │  Entry point and orchestration
└────────┬────────┘
         │
┌────────┴────────┐
│ Core Components │  Business logic
├─────────────────┤
│ • Claude Client │  AI integration
│ • Metadata      │  Data extraction
│ • Organization  │  File organization
│ • File Access   │  Storage abstraction
└────────┬────────┘
         │
┌────────┴────────┐
│    Utilities    │  Supporting functions
└─────────────────┘
```

### Design Patterns

1. **Repository Pattern**: File access layer abstracts storage
2. **Strategy Pattern**: Organization rules and file processors
3. **Factory Pattern**: Component initialization
4. **Observer Pattern**: Progress tracking

## Project Structure

```
file-organizer/
├── src/
│   ├── app.py                    # Main application
│   ├── claude_integration/       # Claude AI integration
│   │   ├── client.py            # API client
│   │   ├── prompts.py           # Prompt templates
│   │   └── rate_limiter.py      # Rate limiting
│   ├── metadata_extraction/      # Metadata processing
│   │   ├── extractor.py         # Main extractor
│   │   ├── storage.py           # Metadata storage
│   │   └── ai_summarizer.py     # AI-powered summarization
│   ├── organization_logic/       # Organization rules
│   │   ├── engine.py            # Rule engine
│   │   ├── rule_manager.py      # Rule management
│   │   └── conflict_resolver.py # Conflict resolution
│   ├── file_access/             # File system abstraction
│   │   ├── local_accessor.py    # Local file access
│   │   ├── dropbox_accessor.py  # Dropbox integration
│   │   ├── processor.py         # File processing
│   │   └── ocr_processor.py     # OCR functionality
│   └── utils/                   # Utility modules
│       ├── config_manager.py    # Configuration
│       ├── batch_processor.py   # Batch processing
│       ├── error_handler.py     # Error handling
│       └── progress.py          # Progress tracking
├── tests/                       # Test suite
├── examples/                    # Example scripts
├── config/                      # Configuration files
└── docs/                       # Documentation
```

## Core Components

### Claude Integration

The Claude integration module handles all AI-related functionality:

```python
# src/claude_integration/client.py
class ClaudeClient:
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def analyze_document(self, content: str, **kwargs) -> AnalysisResult:
        # Document analysis logic
        pass
```

**Extension points:**
- Custom prompt templates
- Additional AI models
- Response parsing strategies

### Metadata Extraction

Extracts structured data from documents:

```python
# src/metadata_extraction/extractor.py
class MetadataExtractor:
    def extract_metadata(self, content: str, file_info: Dict) -> DocumentMetadata:
        # Extraction logic
        pass
```

**Extension points:**
- Custom extraction strategies
- Additional metadata fields
- Specialized document handlers

### Organization Logic

Determines file organization based on rules:

```python
# src/organization_logic/engine.py
class OrganizationEngine:
    def determine_target_location(self, metadata: DocumentMetadata) -> Tuple[str, str]:
        # Rule matching logic
        pass
```

**Extension points:**
- Custom rule types
- Complex conditions
- Alternative organization strategies

### File Access

Abstracts file system operations:

```python
# src/file_access/local_accessor.py
class FileAccessor(ABC):
    @abstractmethod
    def list_files(self, path: str) -> List[FileInfo]:
        pass
    
    @abstractmethod
    def read_file(self, path: str) -> bytes:
        pass
```

**Extension points:**
- New storage backends
- Custom file operations
- Additional file metadata

## Extension Points

### 1. Adding a New Storage Backend

Create a new accessor implementing the FileAccessor interface:

```python
# src/file_access/s3_accessor.py
from src.file_access.base import FileAccessor
import boto3

class S3Accessor(FileAccessor):
    def __init__(self, bucket_name: str, **kwargs):
        self.s3_client = boto3.client('s3', **kwargs)
        self.bucket_name = bucket_name
    
    def list_files(self, prefix: str = "") -> List[FileInfo]:
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        return [self._convert_to_file_info(obj) for obj in response.get('Contents', [])]
    
    def read_file(self, key: str) -> bytes:
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=key
        )
        return response['Body'].read()
```

### 2. Creating Custom Organization Rules

Add specialized rule conditions:

```python
# src/organization_logic/custom_rules.py
class CustomRule(OrganizationRule):
    def matches(self, metadata: DocumentMetadata) -> bool:
        # Custom matching logic
        if metadata.custom_metadata.get('project_id'):
            return True
        return False
    
    def get_path(self, metadata: DocumentMetadata) -> str:
        project_id = metadata.custom_metadata['project_id']
        return f"Projects/{project_id}/{metadata.document_type}"
```

### 3. Adding New Document Processors

Create specialized processors for new file types:

```python
# src/file_access/video_processor.py
from src.file_access.processor import BaseProcessor
import cv2

class VideoProcessor(BaseProcessor):
    def process(self, file_path: str) -> Dict[str, Any]:
        video = cv2.VideoCapture(file_path)
        
        # Extract video metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        
        # Extract thumbnail
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = video.read()
        
        return {
            'duration': duration,
            'fps': fps,
            'thumbnail': frame,
            'metadata': self._extract_metadata(video)
        }
```

### 4. Custom Metadata Extractors

Add domain-specific metadata extraction:

```python
# src/metadata_extraction/medical_extractor.py
class MedicalDocumentExtractor(MetadataExtractor):
    def extract_metadata(self, content: str, file_info: Dict) -> DocumentMetadata:
        metadata = super().extract_metadata(content, file_info)
        
        # Extract medical-specific information
        metadata.custom_metadata['patient_id'] = self._extract_patient_id(content)
        metadata.custom_metadata['diagnosis_codes'] = self._extract_diagnosis_codes(content)
        metadata.custom_metadata['treatment_date'] = self._extract_treatment_date(content)
        
        return metadata
```

## Development Setup

### Setting up the Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/file-organizer.git
cd file-organizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_metadata_extraction.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_extract"
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# or with IPython
import IPython; IPython.embed()
```

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_app.py          # App-level tests
├── test_claude_integration.py
├── test_metadata_extraction.py
├── test_organization_logic.py
├── test_file_access.py
└── test_utils.py
```

### Writing Tests

```python
# tests/test_metadata_extraction.py
import pytest
from src.metadata_extraction.extractor import MetadataExtractor

class TestMetadataExtractor:
    @pytest.fixture
    def extractor(self, mock_claude_client):
        return MetadataExtractor(mock_claude_client)
    
    def test_extract_metadata_invoice(self, extractor):
        content = "Invoice #12345..."
        metadata = extractor.extract_metadata(
            content, 
            file_path="/test/invoice.pdf",
            file_type="pdf"
        )
        
        assert metadata.document_type == "invoice"
        assert "financial" in metadata.categories
```

### Mock Fixtures

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_claude_client():
    client = Mock()
    client.analyze_document.return_value = AnalysisResult(
        content='{"document_type": "invoice"}',
        confidence_score=0.95
    )
    return client
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Update documentation
6. Submit pull request

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(metadata): add support for Excel files`
- `fix(dropbox): handle rate limiting correctly`
- `docs(api): update API reference`
- `test(organization): add edge case tests`

### Code Review Checklist

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No security issues
- [ ] Performance considered
- [ ] Error handling appropriate

## Code Style

### Python Style Guide

Follow PEP 8 with these additions:

```python
# Good: Descriptive names
def extract_document_metadata(file_path: str) -> DocumentMetadata:
    pass

# Bad: Unclear abbreviations
def ext_doc_md(fp: str) -> dict:
    pass
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union

def process_files(
    file_paths: List[str],
    options: Optional[Dict[str, Any]] = None
) -> List[ProcessingResult]:
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def analyze_document(self, content: str, options: Dict[str, Any]) -> AnalysisResult:
    """Analyze document content using Claude AI.
    
    Args:
        content: The document content to analyze
        options: Additional options for analysis
            - custom_prompt: Override default prompt
            - max_tokens: Maximum tokens for response
    
    Returns:
        AnalysisResult containing extracted metadata
    
    Raises:
        APIError: If Claude API request fails
        ValidationError: If content is invalid
    """
    pass
```

### Error Handling

Use specific exceptions:

```python
class MetadataExtractionError(Exception):
    """Raised when metadata extraction fails."""
    pass

class OrganizationError(Exception):
    """Raised when file organization fails."""
    pass
```

### Logging

Use appropriate log levels:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Processing file: %s", file_path)
logger.info("Successfully processed %d files", count)
logger.warning("Rate limit approaching: %d/%d", current, limit)
logger.error("Failed to process file: %s", error)
```

## Best Practices

1. **Dependency Injection**: Pass dependencies explicitly
2. **Single Responsibility**: Each class/function should do one thing
3. **Configuration**: Use configuration files, not hardcoded values
4. **Testing**: Write tests for all new functionality
5. **Documentation**: Update docs with API changes
6. **Security**: Never log sensitive information
7. **Performance**: Profile before optimizing

## Resources

- [Python Best Practices](https://docs.python-guide.org/)
- [Claude API Documentation](https://docs.anthropic.com/)
- [Project Issues](https://github.com/yourusername/file-organizer/issues)
- [Discord Community](https://discord.gg/file-organizer)