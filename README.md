# File Organizer with Claude AI

An intelligent file organization system that uses Claude 3.7 Sonnet AI to analyze and categorize unstructured files, automatically organizing them into logical folders based on extracted metadata.

## 🚀 Features

- **AI-Powered Analysis**: Uses Claude 3.7 Sonnet to intelligently analyze document content
- **Smart Organization**: Automatically categorizes files based on extracted metadata
- **Multi-Format Support**: Handles images (JPG, PNG, TIFF), PDFs, Word documents, and more
- **Batch Processing**: Processes multiple files efficiently with progress tracking
- **Flexible Storage**: Works with local files and Dropbox integration
- **Custom Rules**: Define your own organization rules and folder structures
- **OCR Support**: Extracts text from scanned documents and images
- **Detailed Reports**: Generates comprehensive reports of organization activities

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Development](#-development)
- [License](#-license)

## 🏁 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/file-organizer.git
cd file-organizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration example
cp config/config_example.json config/config.json

# Add your Claude API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run the organizer
python -m src.app --source ./my_files --target ./organized
```

## 📦 Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/file-organizer.git
cd file-organizer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### As a Package

```bash
pip install .
```

For detailed installation instructions, see [INSTALLATION.md](docs/INSTALLATION.md).

## ⚙️ Configuration

The application uses JSON configuration files. Create your configuration by copying the example:

```bash
cp config/config_example.json config/config.json
```

Key configuration sections:
- **API**: Claude API settings and rate limits
- **Source**: Input file locations (local or Dropbox)
- **Processing**: Batch sizes, file types, and OCR settings
- **Organization**: Rules, folder structures, and naming patterns
- **Logging**: Log levels and output locations

For detailed configuration options, see [CONFIGURATION.md](docs/CONFIGURATION.md).

## 🔧 Usage

### Command Line Interface

```bash
# Basic usage
python -m src.app --source /path/to/files --target /path/to/organized

# Dry run (preview without moving files)
python -m src.app --source /path/to/files --dry-run

# Use custom configuration
python -m src.app --config /path/to/config.json

# Verbose output
python -m src.app --source /path/to/files --verbose

# Copy files instead of moving
python -m src.app --source /path/to/files --mode copy
```

### Available Options

- `--config`: Path to configuration file
- `--source`: Source directory or Dropbox path
- `--target`: Target directory for organized files
- `--dry-run`: Preview changes without moving files
- `--mode`: File operation mode (copy or move)
- `--verbose`: Enable verbose output
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

For more usage examples, see [USAGE.md](docs/USAGE.md).

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [Configuration Guide](docs/CONFIGURATION.md) - All configuration options explained
- [Usage Examples](docs/USAGE.md) - Common scenarios and use cases
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Solutions to common issues
- [API Reference](docs/API_REFERENCE.md) - For developers using the package
- [Developer Guide](docs/DEVELOPER.md) - Architecture and extension points

## 💡 Examples

### Organize Photos by Date
```python
from src.organization_logic.engine import OrganizationEngine

engine = OrganizationEngine()
engine.add_rule({
    "name": "Photos by date",
    "conditions": {"document_type": "photo"},
    "path_template": "Photos/{date.year}/{date.month}",
    "priority": 90
})
```

### Process Financial Documents
```python
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor

client = ClaudeClient()
extractor = MetadataExtractor(client)

metadata = extractor.extract_metadata(
    file_content="Invoice content...",
    file_path="/docs/invoice.pdf",
    file_type="pdf"
)
```

For more examples, see the [examples/](examples/) directory.

## 🛠️ Development

### Setting up for development

```bash
# Install development dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

### Code quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Claude AI by Anthropic for intelligent document analysis
- The open-source community for the amazing libraries used in this project
