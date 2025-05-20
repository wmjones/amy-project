# Command Line Interface Reference

The File Organizer provides a comprehensive command-line interface for organizing files using Claude AI.

## Basic Usage

```bash
python -m src.app [OPTIONS]
```

## Command Line Options

### --config

Specifies the path to a configuration file.

- **Type**: String (file path)
- **Default**: `config/config.json`
- **Example**: `--config /path/to/my/config.json`

```bash
python -m src.app --config config/production.json
```

### --source

Specifies the source directory or Dropbox path containing files to organize.

- **Type**: String (directory path)
- **Default**: Current directory or value from config
- **Example**: `--source /home/user/documents`

```bash
python -m src.app --source ~/Downloads
```

For Dropbox paths:
```bash
python -m src.app --source dropbox:/unorganized_files
```

### --target

Specifies the target directory where organized files will be placed.

- **Type**: String (directory path)
- **Default**: `./organized` or value from config
- **Example**: `--target /home/user/organized`

```bash
python -m src.app --target ~/Documents/Organized
```

### --dry-run

Performs a trial run without actually moving or copying files. Shows what would be done.

- **Type**: Boolean flag
- **Default**: False
- **Example**: `--dry-run`

```bash
python -m src.app --source ~/Downloads --dry-run
```

### --mode

Specifies whether to copy or move files during organization.

- **Type**: Choice ['copy', 'move']
- **Default**: 'copy' or value from config
- **Example**: `--mode move`

```bash
python -m src.app --source ~/Downloads --mode move
```

### --verbose

Enables verbose output with detailed processing information.

- **Type**: Boolean flag
- **Default**: False
- **Example**: `--verbose`

```bash
python -m src.app --verbose
```

### --log-level

Sets the logging level for the application.

- **Type**: Choice ['DEBUG', 'INFO', 'WARNING', 'ERROR']
- **Default**: 'INFO'
- **Example**: `--log-level DEBUG`

```bash
python -m src.app --log-level DEBUG
```

## Complete Examples

### Basic file organization
```bash
python -m src.app --source /path/to/files --target /path/to/organized
```

### Dry run with verbose output
```bash
python -m src.app --source ~/Downloads --dry-run --verbose
```

### Move files with custom config
```bash
python -m src.app --config custom.json --source ~/Documents --mode move
```

### Debug mode with all options
```bash
python -m src.app \
  --config config/debug.json \
  --source /data/unorganized \
  --target /data/organized \
  --mode copy \
  --verbose \
  --log-level DEBUG
```

## Configuration Precedence

Options are applied in the following order (later overrides earlier):

1. Default values
2. Configuration file values
3. Command-line arguments

Example:
```bash
# If config.json has mode="move", but CLI specifies --mode copy,
# the files will be copied (CLI takes precedence)
python -m src.app --config config.json --mode copy
```

## Exit Codes

The application returns the following exit codes:

- `0`: Success - all operations completed successfully
- `1`: General error - operation failed
- `2`: Configuration error - invalid configuration
- `3`: Permission error - insufficient permissions
- `4`: API error - Claude API issues

## Environment Variables

The following environment variables can be used:

- `ANTHROPIC_API_KEY`: Claude API key
- `FILE_ORGANIZER_CONFIG`: Default config file path
- `FILE_ORGANIZER_LOG_LEVEL`: Default log level

Example:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export FILE_ORGANIZER_CONFIG=/etc/file-organizer/config.json
python -m src.app --source /data/files
```

## Common Usage Patterns

### Daily organization
```bash
#!/bin/bash
python -m src.app \
  --source ~/Downloads \
  --target ~/Documents/$(date +%Y/%m) \
  --mode move
```

### Batch processing with custom rules
```bash
python -m src.app \
  --config config/custom_rules.json \
  --source /archive/unprocessed \
  --target /archive/organized \
  --verbose
```

### Testing new configuration
```bash
python -m src.app \
  --config new_config.json \
  --source test_files/ \
  --dry-run \
  --verbose
```

## Advanced Options

### Hidden/Development Options

These options are primarily for development and debugging:

- `--validate-config`: Validates configuration without processing
- `--benchmark`: Runs performance benchmarks
- `--profile`: Enables profiling output
- `--trace`: Enables trace-level logging

Example:
```bash
python -m src.app --config test.json --validate-config
```

## Integration with Shell Scripts

The CLI is designed to work well with shell scripts:

```bash
#!/bin/bash
# organize_downloads.sh

SOURCE_DIR="$HOME/Downloads"
TARGET_DIR="$HOME/Documents/Organized"
LOG_FILE="$HOME/logs/organize_$(date +%Y%m%d).log"

# Run organization with error handling
if python -m src.app --source "$SOURCE_DIR" --target "$TARGET_DIR" --mode move >> "$LOG_FILE" 2>&1; then
    echo "Organization completed successfully"
    notify-send "File Organizer" "Downloads organized successfully"
else
    echo "Organization failed - check $LOG_FILE"
    notify-send "File Organizer" "Error organizing downloads"
fi
```

## Tips and Best Practices

1. **Always test with --dry-run first**
   ```bash
   python -m src.app --source important_files/ --dry-run
   ```

2. **Use verbose mode for troubleshooting**
   ```bash
   python -m src.app --source ~/Documents --verbose --log-level DEBUG
   ```

3. **Combine with other Unix tools**
   ```bash
   find ~/Downloads -mtime +30 | python -m src.app --source - --target ~/Archive
   ```

4. **Create aliases for common operations**
   ```bash
   alias organize='python -m src.app --source ~/Downloads --target ~/Organized'
   ```

## Getting Help

To see all available options:
```bash
python -m src.app --help
```

For more detailed documentation, see:
- [USAGE.md](USAGE.md) - Common use cases and examples
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving