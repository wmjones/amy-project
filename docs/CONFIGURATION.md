# Configuration Guide

This guide explains all configuration options available in the File Organizer with Claude AI.

## Configuration File Structure

The application uses JSON configuration files. The main configuration file should be located at `config/config.json`.

```json
{
  "api": {},
  "dropbox": {},
  "source": {},
  "processing": {},
  "organization": {},
  "metadata": {},
  "logging": {},
  "ui": {}
}
```

## Configuration Sections

### API Configuration

Controls Claude AI integration and API settings.

```json
"api": {
  "anthropic_api_key": "YOUR_API_KEY_HERE",
  "claude_model": "claude-3-opus-20240229",
  "rate_limit": 15,
  "max_tokens": 2500,
  "timeout": 45,
  "max_retries": 3,
  "initial_backoff": 2.0
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `anthropic_api_key` | string | required | Your Claude API key (can also use env var `ANTHROPIC_API_KEY`) |
| `claude_model` | string | "claude-3-opus-20240229" | Claude model to use |
| `rate_limit` | integer | 15 | Maximum requests per minute |
| `max_tokens` | integer | 2500 | Maximum tokens per request |
| `timeout` | integer | 45 | Request timeout in seconds |
| `max_retries` | integer | 3 | Number of retry attempts |
| `initial_backoff` | float | 2.0 | Initial backoff delay for retries |

### Dropbox Configuration

Settings for Dropbox integration (optional).

```json
"dropbox": {
  "app_key": "YOUR_DROPBOX_APP_KEY",
  "app_secret": "YOUR_DROPBOX_APP_SECRET",
  "access_token": null,
  "refresh_token": null,
  "source_folder": "/file-organizer-uploads",
  "download_batch_size": 20
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `app_key` | string | required | Dropbox app key |
| `app_secret` | string | required | Dropbox app secret |
| `access_token` | string | null | Dropbox access token |
| `refresh_token` | string | null | Dropbox refresh token |
| `source_folder` | string | "/" | Dropbox folder to process |
| `download_batch_size` | integer | 20 | Files to download per batch |

### Source Configuration

Defines where to read files from.

```json
"source": {
  "type": "local",
  "directory": "./input_files"
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | string | "local" | Source type: "local" or "dropbox" |
| `directory` | string | "./" | Local directory path (when type is "local") |

### Processing Configuration

Controls file processing behavior.

```json
"processing": {
  "batch_size": 15,
  "max_workers": 6,
  "file_types": [
    ".jpg", ".jpeg", ".png", ".tiff", ".bmp",
    ".pdf", ".docx", ".doc", ".txt", ".csv"
  ],
  "max_file_size": 104857600,
  "queue_priority": "size",
  "enable_ocr": true,
  "ocr_languages": ["eng", "fra", "deu"]
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | integer | 10 | Number of files to process per batch |
| `max_workers` | integer | 4 | Maximum concurrent workers |
| `file_types` | array | [common formats] | Supported file extensions |
| `max_file_size` | integer | 100MB | Maximum file size in bytes |
| `queue_priority` | string | "fifo" | Queue priority: "fifo", "size", "type" |
| `enable_ocr` | boolean | true | Enable OCR for images |
| `ocr_languages` | array | ["eng"] | OCR language codes |

### Organization Configuration

Defines how files are organized.

```json
"organization": {
  "mode": "copy",
  "base_directory": "./organized_files",
  "conflict_resolution": "rename",
  "preserve_structure": false,
  "use_default_rules": true,
  "default_folder": "Unsorted",
  "rules": []
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | "copy" | File operation: "copy" or "move" |
| `base_directory` | string | "./organized" | Output directory |
| `conflict_resolution` | string | "rename" | How to handle conflicts: "rename", "skip", "overwrite" |
| `preserve_structure` | boolean | false | Preserve source directory structure |
| `use_default_rules` | boolean | true | Use built-in organization rules |
| `default_folder` | string | "Unsorted" | Folder for uncategorized files |
| `rules` | array | [] | Custom organization rules |

#### Organization Rules

Custom rules for file organization:

```json
"rules": [
  {
    "name": "Invoices by client and year",
    "priority": 1,
    "conditions": {
      "document_type": "invoice",
      "entities.organizations": {"$exists": true}
    },
    "path_template": "Financial/Invoices/{entities.organizations[0]}/{dates.document_date|%Y}/{filename}",
    "enabled": true
  }
]
```

Rule structure:
- `name`: Rule description
- `priority`: Execution priority (lower numbers first)
- `conditions`: Matching conditions
- `path_template`: Output path template
- `enabled`: Enable/disable rule

Path template variables:
- `{filename}`: Original filename
- `{document_type}`: Detected document type
- `{entities.organizations[0]}`: First organization entity
- `{dates.document_date|%Y}`: Document date (year)
- `{tags[0]}`: First tag

### Metadata Configuration

Settings for metadata storage.

```json
"metadata": {
  "storage_backend": "sqlite",
  "db_path": "./data/metadata.db",
  "json_path": "./data/metadata.json",
  "index_fields": [
    "document_type",
    "dates.document_date",
    "entities.people",
    "entities.organizations"
  ]
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `storage_backend` | string | "json" | Storage type: "json" or "sqlite" |
| `db_path` | string | "./metadata.db" | SQLite database path |
| `json_path` | string | "./metadata.json" | JSON file path |
| `index_fields` | array | [] | Fields to index for search |

### Logging Configuration

Controls application logging.

```json
"logging": {
  "level": "INFO",
  "file": "./logs/file_organizer.log",
  "max_size": 10485760,
  "backup_count": 5,
  "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | "INFO" | Log level: DEBUG, INFO, WARNING, ERROR |
| `file` | string | "./logs/app.log" | Log file path |
| `max_size` | integer | 10MB | Maximum log file size |
| `backup_count` | integer | 5 | Number of backup files |
| `format` | string | standard | Log message format |

### UI Configuration

User interface settings.

```json
"ui": {
  "show_progress": true,
  "update_interval": 2,
  "color_output": true
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `show_progress` | boolean | true | Show progress bars |
| `update_interval` | integer | 1 | Progress update interval (seconds) |
| `color_output` | boolean | true | Enable colored output |

## Environment Variables

Environment variables can override configuration file settings:

| Variable | Config Path | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | `api.anthropic_api_key` | Claude API key |
| `FILE_ORGANIZER_CONFIG` | - | Path to config file |
| `FILE_ORGANIZER_LOG_LEVEL` | `logging.level` | Log level |

## Configuration Precedence

Configuration values are loaded in this order (later overrides earlier):
1. Default values in code
2. Configuration file
3. Environment variables
4. Command-line arguments

## Example Configurations

### Minimal Configuration

```json
{
  "api": {
    "anthropic_api_key": "sk-ant-..."
  },
  "source": {
    "type": "local",
    "directory": "./my_files"
  },
  "organization": {
    "base_directory": "./organized"
  }
}
```

### Dropbox Integration

```json
{
  "api": {
    "anthropic_api_key": "sk-ant-..."
  },
  "source": {
    "type": "dropbox"
  },
  "dropbox": {
    "app_key": "your_app_key",
    "app_secret": "your_app_secret",
    "source_folder": "/uploads"
  }
}
```

### High-Volume Processing

```json
{
  "api": {
    "anthropic_api_key": "sk-ant-...",
    "rate_limit": 30,
    "max_tokens": 1500
  },
  "processing": {
    "batch_size": 25,
    "max_workers": 8,
    "max_file_size": 52428800
  },
  "logging": {
    "level": "WARNING"
  }
}
```

### Custom Organization Rules

```json
{
  "organization": {
    "use_default_rules": false,
    "rules": [
      {
        "name": "Project documents",
        "priority": 1,
        "conditions": {
          "tags": {"$contains": "project"}
        },
        "path_template": "Projects/{tags[0]}/{document_type}"
      },
      {
        "name": "Date-based photos",
        "priority": 2,
        "conditions": {
          "document_type": "photo"
        },
        "path_template": "Photos/{dates.taken_date|%Y/%m/%d}"
      }
    ]
  }
}
```

## Validation

The application validates configuration on startup. Common validation errors:

- Missing required fields (e.g., API key)
- Invalid file paths
- Unsupported file types
- Invalid rule syntax

To validate your configuration:
```bash
python -m src.app --config /path/to/config.json --validate-config
```

## Best Practices

1. **Store sensitive data in environment variables** rather than config files
2. **Use version control** for configuration files (exclude API keys)
3. **Start with defaults** and customize as needed
4. **Test rules** with dry-run mode before processing
5. **Set appropriate rate limits** to avoid API throttling
6. **Configure logging** for production environments
7. **Regular backups** of metadata databases

## Migration

When upgrading from older versions:

1. Backup existing configuration
2. Compare with new example configuration
3. Migrate custom rules and settings
4. Test with a small batch first

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common configuration issues.