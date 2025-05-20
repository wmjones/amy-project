# CLAUDE.local.md

## Project Directory Structure

```
/workspaces/amy-project/
   .devcontainer/          # Development container configuration
   .claude/                # Claude-specific configuration
   config/                 # Application configuration files
   data/                   # Static/sample data only
      hansman_samples/     # Sample Hansman documents
   docs/                   # Project documentation
   examples/               # Demo scripts and examples
   scripts/                # Standalone scripts
      processing/          # Data processing scripts
      setup/               # Setup and configuration scripts
      testing/             # Test and debugging scripts
      evaluation/          # Analysis and evaluation scripts
   src/                    # Application source code
      app.py              # Main application entry point
      claude_integration/  # Claude API integration
      file_access/         # File system and Dropbox access
      metadata_extraction/ # Metadata and AI extraction
      organization_logic/  # File organization engine
      optimization/        # Performance optimization
      utils/              # Utility modules
   tasks/                  # Task Master project management
   tests/                  # Unit tests
   workspace/              # All working data and outputs
      downloads/           # Downloaded files (cached)
      processed/           # Processed/organized data
      reports/             # Generated reports
      cache/              # Temporary cache files
      temp/               # Temporary working files
```

## User Preferences

- **IMPORTANT**: Always search the web for simple solutions or available packages before implementing custom code
- Prefer to first use context7 to get documentation and use the web as a fallback
- Use Task Master to manage tasks and assume the Task Master project has been initialized
- Prefer importing and using existing functionality over creating custom implementations
- Look for established Python packages, especially for OCR, image processing, and file handling
- Only create custom code when necessary or when existing solutions don't meet requirements
- Prioritize well-maintained, popular packages with good documentation
- Keep solutions simple and maintainable by leveraging community packages

## Project Context

- Python-based document processing and organization application
- Integrates with Dropbox API for file access and organization
- Uses Claude AI for document analysis and metadata extraction
- OCR capabilities for processing scanned documents
- Batch processing for handling large document collections
- Focus on Hansman historical document archive processing
- Task Master for project management
- Modular architecture with clear separation of concerns

## Working with Data

- Always use `workspace/` directory for data operations
- Downloaded files are cached in `workspace/downloads/` to avoid re-downloading
- Processed results go to `workspace/processed/`
- Reports are generated in `workspace/reports/`
- Use `src/utils/paths.py` for consistent path handling
- The `data/` directory contains only static sample files

## Key Integration Points

- Dropbox API for cloud file access
- Claude AI for document understanding
- Tesseract OCR for text extraction
- Python PIL/Pillow for image processing
- Task Master for project tracking
