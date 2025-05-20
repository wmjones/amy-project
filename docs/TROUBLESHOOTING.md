# Troubleshooting Guide

This guide helps you resolve common issues with the File Organizer.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [API and Authentication](#api-and-authentication)
- [File Processing Errors](#file-processing-errors)
- [Performance Issues](#performance-issues)
- [Dropbox Integration](#dropbox-integration)
- [OCR Problems](#ocr-problems)
- [Common Error Messages](#common-error-messages)

## Installation Issues

### ImportError: No module named 'anthropic'

**Problem**: Missing dependencies after installation.

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Visual C++ error on Windows

**Problem**: Compilation errors when installing certain packages.

**Solution**:
1. Install Visual C++ Build Tools from Microsoft
2. Or use pre-compiled wheels:
   ```bash
   pip install --only-binary :all: -r requirements.txt
   ```

### Permission denied during installation

**Problem**: Cannot install packages due to permissions.

**Solution**:
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix virtual environment permissions
chmod -R 755 venv/
```

## Configuration Problems

### Configuration file not found

**Problem**: Application cannot find config.json.

**Solution**:
```bash
# Create from example
cp config/config_example.json config/config.json

# Or specify path explicitly
python -m src.app --config /path/to/config.json
```

### Invalid JSON in configuration

**Problem**: JSON parsing error in config file.

**Solution**:
1. Validate JSON syntax:
   ```bash
   python -m json.tool config/config.json
   ```
2. Check for common issues:
   - Missing commas between items
   - Trailing commas after last item
   - Unmatched brackets or quotes

### Configuration values not applying

**Problem**: Changes to config file don't take effect.

**Solution**:
- Ensure you're editing the correct file
- Check command-line arguments (they override config)
- Verify JSON structure matches expected format

## API and Authentication

### ANTHROPIC_API_KEY not found

**Problem**: Claude API key is not configured.

**Solution**:
```bash
# Set environment variable
export ANTHROPIC_API_KEY=sk-ant-...

# Or add to config.json
{
  "api": {
    "anthropic_api_key": "sk-ant-..."
  }
}
```

### API rate limit exceeded

**Problem**: Too many requests to Claude API.

**Solution**:
1. Reduce rate limit in config:
   ```json
   {
     "api": {
       "rate_limit": 10
     }
   }
   ```
2. Implement retry logic:
   ```json
   {
     "api": {
       "max_retries": 5,
       "initial_backoff": 2.0
     }
   }
   ```

### Invalid API key

**Problem**: Authentication fails with Claude API.

**Solution**:
- Verify API key is correct
- Check key hasn't expired
- Ensure no extra spaces or characters
- Test key with minimal example:
  ```python
  import anthropic
  client = anthropic.Anthropic(api_key="your-key")
  ```

## File Processing Errors

### File not found errors

**Problem**: Application can't access source files.

**Solution**:
- Use absolute paths instead of relative
- Check file permissions
- Verify paths don't contain special characters
- On Windows, use forward slashes or raw strings

### Permission denied when moving files

**Problem**: Cannot write to target directory.

**Solution**:
```bash
# Check directory permissions
ls -la /target/directory

# Fix permissions
chmod 755 /target/directory

# Or run with appropriate user
sudo -u correct_user python -m src.app ...
```

### Memory error with large files

**Problem**: Out of memory when processing large files.

**Solution**:
1. Set file size limits:
   ```json
   {
     "processing": {
       "max_file_size": 52428800
     }
   }
   ```
2. Reduce batch size:
   ```json
   {
     "processing": {
       "batch_size": 5
     }
   }
   ```

## Performance Issues

### Slow processing speed

**Problem**: Files taking too long to process.

**Solution**:
1. Increase batch size and workers:
   ```json
   {
     "processing": {
       "batch_size": 25,
       "max_workers": 8
     }
   }
   ```
2. Optimize API usage:
   ```json
   {
     "api": {
       "max_tokens": 1500
     }
   }
   ```

### High memory usage

**Problem**: Application using too much RAM.

**Solution**:
- Process fewer files at once
- Disable caching for large files
- Use streaming for large documents

### CPU throttling

**Problem**: System becomes unresponsive during processing.

**Solution**:
```json
{
  "processing": {
    "max_workers": 2,
    "cpu_limit": 50
  }
}
```

## Dropbox Integration

### Dropbox authentication failed

**Problem**: Cannot connect to Dropbox account.

**Solution**:
1. Regenerate app credentials
2. Use OAuth flow to get new tokens
3. Check app permissions in Dropbox settings

### Dropbox rate limiting

**Problem**: Too many Dropbox API requests.

**Solution**:
```json
{
  "dropbox": {
    "download_batch_size": 10,
    "request_delay": 1.0
  }
}
```

### File not found in Dropbox

**Problem**: Specified Dropbox path doesn't exist.

**Solution**:
- Verify path format (starts with /)
- Check case sensitivity
- Ensure app has folder access

## OCR Problems

### Tesseract not found

**Problem**: OCR functionality not working.

**Solution**:
```bash
# Install Tesseract
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify installation
tesseract --version
```

### Poor OCR quality

**Problem**: Text extraction is inaccurate.

**Solution**:
1. Preprocess images:
   ```json
   {
     "processing": {
       "ocr_preprocess": true
     }
   }
   ```
2. Use appropriate languages:
   ```json
   {
     "processing": {
       "ocr_languages": ["eng", "fra"]
     }
   }
   ```

### OCR timeout

**Problem**: OCR takes too long for complex images.

**Solution**:
```json
{
  "processing": {
    "ocr_timeout": 60,
    "ocr_max_size": 10485760
  }
}
```

## Common Error Messages

### "No files found to process"

**Causes**:
- Wrong source directory
- No files match configured types
- Permission issues

**Solution**:
```bash
# Check directory contents
ls -la /source/directory

# Verify file types in config
# Add more types if needed
```

### "API request failed: 429"

**Cause**: Rate limit exceeded

**Solution**:
- Wait and retry
- Reduce rate limit in config
- Implement exponential backoff

### "Failed to determine organization path"

**Causes**:
- No matching organization rules
- Metadata extraction failed

**Solution**:
- Add fallback rules
- Check metadata extraction logs
- Use default folder for unmatched files

### "Database locked"

**Cause**: Concurrent access to metadata database

**Solution**:
```json
{
  "metadata": {
    "storage_backend": "json",
    "enable_locking": true
  }
}
```

## Debug Mode

Enable debug mode for detailed diagnostics:

```bash
python -m src.app --log-level DEBUG --verbose
```

Debug configuration:
```json
{
  "logging": {
    "level": "DEBUG",
    "enable_trace": true
  }
}
```

## Getting Help

If issues persist:

1. Check the logs:
   ```bash
   tail -f logs/file_organizer.log
   ```

2. Run validation:
   ```bash
   python -m src.app --validate-config
   ```

3. Test with minimal config:
   ```json
   {
     "api": {
       "anthropic_api_key": "your-key"
     }
   }
   ```

4. Report issues on GitHub with:
   - Error messages
   - Configuration (without sensitive data)
   - System information
   - Steps to reproduce

## Common Fixes Summary

1. **Always check**:
   - API key is set correctly
   - Paths are absolute
   - Permissions are correct
   - JSON syntax is valid

2. **For performance**:
   - Adjust batch sizes
   - Limit file sizes
   - Use appropriate worker counts

3. **For errors**:
   - Enable debug logging
   - Try with smaller datasets
   - Test individual components

4. **For integration issues**:
   - Verify credentials
   - Check API limits
   - Test connections independently