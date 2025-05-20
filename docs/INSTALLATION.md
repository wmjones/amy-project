# Installation Guide

This guide provides detailed instructions for installing the File Organizer with Claude AI on various systems.

## Prerequisites

Before installing, ensure you have:

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- An active Claude API key from Anthropic

### System-specific requirements

#### Windows
- Visual C++ Build Tools (for some dependencies)
- Windows Terminal or PowerShell (recommended)

#### macOS
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew (optional but recommended)

#### Linux
- gcc/g++ compiler
- python3-dev package
- tesseract-ocr (for OCR functionality)

## Installation Methods

### Method 1: Install from Source (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/file-organizer.git
   cd file-organizer
   ```

2. **Create a virtual environment**
   ```bash
   # On Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies (optional)**
   ```bash
   pip install -e ".[dev]"
   ```

### Method 2: Install as a Package

1. **Clone and navigate to the repository**
   ```bash
   git clone https://github.com/yourusername/file-organizer.git
   cd file-organizer
   ```

2. **Install the package**
   ```bash
   pip install .
   ```

   Or for development:
   ```bash
   pip install -e .
   ```

## Configuration

1. **Copy the example configuration**
   ```bash
   cp config/config_example.json config/config.json
   ```

2. **Set up your API key**
   
   Option A: Environment variable (recommended)
   ```bash
   # Linux/macOS
   export ANTHROPIC_API_KEY=your_api_key_here
   
   # Windows
   set ANTHROPIC_API_KEY=your_api_key_here
   ```
   
   Option B: Add to config.json
   ```json
   {
     "api": {
       "anthropic_api_key": "your_api_key_here"
     }
   }
   ```

3. **Configure Dropbox (optional)**
   
   If using Dropbox integration:
   - Create a Dropbox app at https://www.dropbox.com/developers
   - Add app credentials to config.json
   - See [DROPBOX_SETUP.md](DROPBOX_SETUP.md) for detailed instructions

## Installing Additional Components

### OCR Support (Tesseract)

For processing scanned documents:

#### Linux
```bash
sudo apt-get install tesseract-ocr
# For additional languages
sudo apt-get install tesseract-ocr-[lang]
```

#### macOS
```bash
brew install tesseract
# For additional languages
brew install tesseract-lang
```

#### Windows
1. Download the installer from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and follow instructions
3. Add Tesseract to your PATH

### poppler-utils (for PDF processing)

#### Linux
```bash
sudo apt-get install poppler-utils
```

#### macOS
```bash
brew install poppler
```

#### Windows
1. Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract and add to PATH

## Verification

1. **Verify Python installation**
   ```bash
   python --version
   ```
   Should show Python 3.9 or higher

2. **Verify package installation**
   ```bash
   pip show file-organizer
   ```
   
3. **Test the installation**
   ```bash
   python -m src.app --help
   ```
   Should display the help message

4. **Run a test process**
   ```bash
   python -m src.app --source ./examples/sample_files --dry-run
   ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'anthropic'**
   - Ensure you've activated your virtual environment
   - Run `pip install -r requirements.txt` again

2. **API Key not found**
   - Check if ANTHROPIC_API_KEY is set: `echo $ANTHROPIC_API_KEY`
   - Verify the key is correct in config.json

3. **Permission denied errors**
   - Ensure you have write permissions in the output directory
   - On Unix systems, you may need to use `sudo` for system directories

4. **OCR not working**
   - Verify Tesseract is installed: `tesseract --version`
   - Check the path is correctly set

### Platform-specific Issues

#### Windows
- If you get compiler errors, install Visual C++ Build Tools
- Use forward slashes (/) in paths or escape backslashes (\\\\)

#### macOS
- If you get SSL errors, update certificates: `pip install --upgrade certifi`
- For M1/M2 Macs, some dependencies may need Rosetta

#### Linux
- Install python3-dev if you get compilation errors
- Some distributions may need additional system packages

## Updating

To update to the latest version:

```bash
cd file-organizer
git pull origin main
pip install -r requirements.txt --upgrade
```

If installed as a package:
```bash
pip install --upgrade .
```

## Uninstalling

1. **Deactivate virtual environment**
   ```bash
   deactivate
   ```

2. **Remove the virtual environment**
   ```bash
   # Linux/macOS
   rm -rf venv/
   
   # Windows
   rmdir /s venv
   ```

3. **If installed as a package**
   ```bash
   pip uninstall file-organizer
   ```

## Next Steps

Once installation is complete:
1. Review the [CONFIGURATION.md](CONFIGURATION.md) guide
2. Check out [USAGE.md](USAGE.md) for examples
3. Run your first organization task

For any issues not covered here, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.