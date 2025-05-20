#!/usr/bin/env python3
"""
Complete workflow demo showing end-to-end file organization process.
This demonstrates how to use the File Organizer programmatically.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.app import FileOrganizerApp
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor
from src.organization_logic.engine import OrganizationEngine
from src.file_access.manipulator import FileManipulator
from src.utils.progress import ProgressTracker
from src.utils.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_files(demo_dir: Path):
    """Create sample files for demonstration."""
    logger.info("Creating demo files...")
    
    # Create various document types
    files = {
        "invoice_2024_001.txt": """
        INVOICE #2024-001
        Date: March 15, 2024
        
        Bill To: Acme Corporation
        
        Services Rendered:
        - Consulting: $5,000
        - Development: $15,000
        
        Total: $20,000
        Due Date: April 15, 2024
        """,
        
        "contract_johndoe.txt": """
        EMPLOYMENT CONTRACT
        
        This agreement is made on January 1, 2024 between:
        Employer: TechCorp Inc.
        Employee: John Doe
        
        Position: Senior Developer
        Start Date: January 15, 2024
        Salary: $120,000 per year
        """,
        
        "meeting_notes_20240320.txt": """
        Meeting Notes - Project Alpha
        Date: March 20, 2024
        
        Attendees:
        - Jane Smith (Project Manager)
        - Bob Johnson (Developer)
        - Alice Brown (Designer)
        
        Topics Discussed:
        1. Q1 milestone review
        2. Budget allocation
        3. Next sprint planning
        
        Action Items:
        - Bob: Complete API integration by March 25
        - Alice: Finalize UI mockups by March 22
        """,
        
        "medical_report.txt": """
        Medical Report
        
        Patient: John Doe
        Date: February 28, 2024
        Doctor: Dr. Sarah Wilson
        
        Diagnosis: Annual checkup - all results normal
        Next appointment: February 2025
        """,
        
        "receipt_store_20240315.txt": """
        RECEIPT
        
        Store: TechMart
        Date: March 15, 2024
        
        Items:
        - USB Drive: $25.99
        - Keyboard: $89.99
        - Mouse: $45.00
        
        Total: $160.98
        Payment: Credit Card
        """
    }
    
    for filename, content in files.items():
        file_path = demo_dir / filename
        file_path.write_text(content)
        logger.info(f"Created: {filename}")
    
    return list(files.keys())


def demo_complete_workflow():
    """Demonstrate the complete file organization workflow."""
    print("=== Complete File Organization Workflow Demo ===\n")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        target_dir = temp_path / "organized"
        source_dir.mkdir()
        target_dir.mkdir()
        
        # Create demo files
        created_files = create_demo_files(source_dir)
        print(f"Created {len(created_files)} demo files in {source_dir}\n")
        
        # Step 1: Configure the application
        print("Step 1: Configuring the application...")
        config = {
            "api": {
                "anthropic_api_key": "demo_key",  # Would use real key in production
                "claude_model": "claude-3-opus-20240229",
                "rate_limit": 10
            },
            "source": {
                "type": "local",
                "directory": str(source_dir)
            },
            "processing": {
                "batch_size": 5,
                "file_types": [".txt", ".pdf", ".docx"]
            },
            "organization": {
                "mode": "copy",
                "base_directory": str(target_dir),
                "use_default_rules": False,
                "rules": [
                    {
                        "name": "Financial documents",
                        "priority": 1,
                        "conditions": {
                            "document_type": {"$in": ["invoice", "receipt"]}
                        },
                        "path_template": "Financial/{document_type}/{dates.document_date|%Y}/{filename}"
                    },
                    {
                        "name": "Legal documents",
                        "priority": 2,
                        "conditions": {
                            "document_type": "contract"
                        },
                        "path_template": "Legal/Contracts/{entities.people[0]}/{dates.document_date|%Y}/{filename}"
                    },
                    {
                        "name": "Meeting notes",
                        "priority": 3,
                        "conditions": {
                            "document_type": "meeting_minutes"
                        },
                        "path_template": "Meetings/{dates.document_date|%Y}/{dates.document_date|%m}-{dates.document_date|%B}/{filename}"
                    },
                    {
                        "name": "Medical records",
                        "priority": 4,
                        "conditions": {
                            "categories": {"$contains": "medical"}
                        },
                        "path_template": "Medical/{entities.people[0]}/{dates.document_date|%Y}/{filename}"
                    }
                ]
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        # Save config
        config_file = temp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_file}\n")
        
        # Step 2: Initialize the application (mock mode for demo)
        print("Step 2: Initializing the application...")
        
        # For demo purposes, we'll create components manually
        # In production, this would be done by FileOrganizerApp
        
        # Mock Claude client
        from unittest.mock import Mock
        mock_claude = Mock()
        
        # Configure mock responses for different file types
        def mock_analyze(content, file_name, file_type, **kwargs):
            result = Mock()
            
            if "invoice" in file_name.lower():
                result.content = json.dumps({
                    "document_type": "invoice",
                    "categories": ["financial"],
                    "dates": {"document_date": "2024-03-15"},
                    "entities": {
                        "organizations": ["Acme Corporation"],
                        "amounts": {"total": 20000}
                    },
                    "summary": "Invoice for consulting and development services"
                })
            elif "contract" in file_name.lower():
                result.content = json.dumps({
                    "document_type": "contract",
                    "categories": ["legal", "employment"],
                    "dates": {"document_date": "2024-01-01"},
                    "entities": {
                        "people": ["John Doe"],
                        "organizations": ["TechCorp Inc"]
                    },
                    "summary": "Employment contract for senior developer position"
                })
            elif "meeting" in file_name.lower():
                result.content = json.dumps({
                    "document_type": "meeting_minutes",
                    "categories": ["business", "project"],
                    "dates": {"document_date": "2024-03-20"},
                    "entities": {
                        "people": ["Jane Smith", "Bob Johnson", "Alice Brown"]
                    },
                    "summary": "Project Alpha Q1 milestone review meeting"
                })
            elif "medical" in file_name.lower():
                result.content = json.dumps({
                    "document_type": "medical_report",
                    "categories": ["medical", "health"],
                    "dates": {"document_date": "2024-02-28"},
                    "entities": {
                        "people": ["John Doe", "Dr. Sarah Wilson"]
                    },
                    "summary": "Annual checkup report - all results normal"
                })
            elif "receipt" in file_name.lower():
                result.content = json.dumps({
                    "document_type": "receipt",
                    "categories": ["financial"],
                    "dates": {"document_date": "2024-03-15"},
                    "entities": {
                        "organizations": ["TechMart"],
                        "amounts": {"total": 160.98}
                    },
                    "summary": "Purchase receipt for tech equipment"
                })
            
            result.confidence_score = 0.92
            result.tokens_used = 150
            result.metadata = {}
            result.model = "claude-3-opus"
            
            return result
        
        mock_claude.analyze_document = mock_analyze
        
        # Initialize components
        extractor = MetadataExtractor(mock_claude)
        engine = OrganizationEngine(use_default_rules=False)
        
        # Add custom rules
        for rule in config["organization"]["rules"]:
            engine.add_rule(rule)
        
        manipulator = FileManipulator(
            base_directory=target_dir,
            dry_run=False
        )
        
        progress_tracker = ProgressTracker(len(created_files))
        
        print("Application initialized successfully\n")
        
        # Step 3: Process files
        print("Step 3: Processing files...")
        
        processed_count = 0
        error_count = 0
        
        for filename in created_files:
            file_path = source_dir / filename
            print(f"\nProcessing: {filename}")
            
            try:
                # Read file content
                content = file_path.read_text()
                
                # Extract metadata
                print("  - Extracting metadata...")
                metadata = extractor.extract_metadata(
                    file_content=content,
                    file_path=str(file_path),
                    file_type="txt"
                )
                
                print(f"  - Document type: {metadata.document_type}")
                print(f"  - Categories: {', '.join(metadata.categories)}")
                
                # Determine organization
                print("  - Determining organization...")
                target_path, rule_name = engine.determine_target_location(metadata)
                print(f"  - Using rule: {rule_name}")
                print(f"  - Target path: {target_path}")
                
                # Copy file
                print("  - Copying file...")
                success = manipulator.organize_file(
                    str(file_path),
                    target_path,
                    "copy"
                )
                
                if success:
                    print("  ✓ Successfully organized")
                    processed_count += 1
                    progress_tracker.update_progress(
                        filename,
                        "success",
                        {"target": target_path}
                    )
                else:
                    print("  ✗ Failed to organize")
                    error_count += 1
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                error_count += 1
                progress_tracker.update_progress(
                    filename,
                    "error",
                    str(e)
                )
        
        print(f"\n\nProcessing complete!")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors: {error_count}")
        
        # Step 4: Generate report
        print("\nStep 4: Generating report...")
        
        report_generator = ReportGenerator(progress_tracker, manipulator)
        report = report_generator.generate_summary_report()
        
        # Save report
        report_file = target_dir / "organization_report.txt"
        report_file.write_text(report)
        print(f"Report saved to: {report_file}")
        
        # Step 5: Show organized structure
        print("\nStep 5: Final organized structure:")
        
        def show_tree(directory, prefix=""):
            """Display directory tree."""
            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    show_tree(item, prefix + extension)
        
        show_tree(target_dir)
        
        # Step 6: Verify organization
        print("\n\nStep 6: Verification")
        print("Checking organized files...")
        
        for root, dirs, files in target_dir.walk():
            for file in files:
                if file.endswith('.txt'):
                    rel_path = (root / file).relative_to(target_dir)
                    print(f"  ✓ {rel_path}")
        
        print("\n=== Demo Complete ===")
        
        # Keep temp directory for inspection
        input("\nPress Enter to clean up and exit...")


def demo_custom_processor():
    """Demonstrate creating a custom file processor."""
    print("\n=== Custom File Processor Demo ===\n")
    
    # Example of extending the file processor
    from src.file_access.processor import FileProcessor
    
    class CustomProcessor(FileProcessor):
        """Custom processor with additional capabilities."""
        
        def process_custom_format(self, file_path: str) -> str:
            """Process a custom file format."""
            # Custom processing logic here
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Apply custom transformations
            processed = content.upper()  # Example transformation
            
            return processed
    
    # Use the custom processor
    processor = CustomProcessor()
    print("Created custom processor with extended capabilities")
    print("This processor can handle standard formats plus custom ones")


if __name__ == "__main__":
    # Run the complete workflow demo
    demo_complete_workflow()
    
    # Show custom processor example
    print("\n" + "="*50 + "\n")
    demo_custom_processor()