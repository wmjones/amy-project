# Amy Project Documentation Navigator

I'll help you understand the Amy Project documentation structure based on your area of interest: $ARGUMENTS

## Core Files - Always Recommended

Let me search for and provide summaries of these essential documentation files:

1. **Main README** - `/workspaces/amy-project/README.md`
2. **Installation Guide** - `/workspaces/amy-project/docs/INSTALLATION.md`
3. **Configuration Guide** - `/workspaces/amy-project/docs/CONFIGURATION.md`
4. **API Reference** - `/workspaces/amy-project/docs/API_REFERENCE.md`

## Specific Documentation Based on Your Focus

{{#if_eq $ARGUMENTS "dropbox"}}
### Dropbox Integration Documentation
I'll prioritize analyzing these files for you:
1. **Dropbox Integration Overview** - `/workspaces/amy-project/docs/dropbox_integration.md`
2. **Dropbox Setup Guide** - `/workspaces/amy-project/docs/DROPBOX_SETUP.md`
3. **Dropbox App Setup** - `/workspaces/amy-project/docs/DROPBOX_APP_SETUP.md`
4. **Authentication Guide** - `/workspaces/amy-project/docs/AUTHENTICATION_REQUIRED.md`
{{/if_eq}}

{{#if_eq $ARGUMENTS "ocr"}}
### OCR Processing Documentation
I'll prioritize analyzing these files for you:
1. **OCR Processor Module** - `/workspaces/amy-project/src/file_access/ocr_processor.py`
2. **OCR AI Pipeline** - `/workspaces/amy-project/src/metadata_extraction/ocr_ai_pipeline.py`
3. **OCR Evaluation Scripts** - `/workspaces/amy-project/scripts/evaluation/ocr_evaluation.py`
{{/if_eq}}

{{#if_eq $ARGUMENTS "ai"}}
### AI and Claude Integration Documentation
I'll prioritize analyzing these files for you:
1. **Claude Integration Module** - `/workspaces/amy-project/src/claude_integration/client.py`
2. **AI Summarizer** - `/workspaces/amy-project/src/metadata_extraction/ai_summarizer.py`
3. **Claude Demo** - `/workspaces/amy-project/examples/claude_demo.py`
{{/if_eq}}

{{#if_eq $ARGUMENTS "hansman"}}
### Hansman Processing Documentation
I'll prioritize analyzing these files for you:
1. **Hansman Setup Guide** - `/workspaces/amy-project/SETUP_GUIDE_HANSMAN_FOLDER.md`
2. **Hansman Processing Scripts** - `/workspaces/amy-project/scripts/processing/process_hansman_*.py`
3. **Hansman POC** - `/workspaces/amy-project/src/proof_of_concept/hansman_ocr_poc.py`
{{/if_eq}}

## Implementation Documentation
I'll also examine how everything connects:
- **Implementation Plan** - `/workspaces/amy-project/docs/implementation_plan_and_cost_analysis.md`
- **Implementation Summary** - `/workspaces/amy-project/docs/implementation_summary.md`
- **Project Structure** - `/workspaces/amy-project/docs/PROJECT_STRUCTURE_RECOMMENDATION.md`

## Analysis Instructions
After reading these files, I will:
1. Provide a concise summary of the documentation structure
2. Highlight key components relevant to your focus area
3. Recommend a logical reading sequence
4. Answer any specific questions you have about the documentation

## Usage Examples
- `amy-docs dropbox` - Focus on Dropbox integration documentation
- `amy-docs ocr` - Focus on OCR processing documentation
- `amy-docs ai` - Focus on AI and Claude integration
- `amy-docs hansman` - Focus on Hansman-specific processing
- `amy-docs` - General documentation overview