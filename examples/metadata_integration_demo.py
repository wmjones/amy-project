#!/usr/bin/env python3
"""
Demo script for metadata integration between OCR + AI pipeline and existing systems.
Shows how the new AI-driven metadata enriches and integrates with legacy metadata.
"""

import sys
import os
from pathlib import Path
import json
import logging
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.file_access.ocr_processor import OCRProcessor, OCRResult
from src.metadata_extraction.ai_summarizer import AISummarizer, DocumentSummary
from src.metadata_extraction.ocr_ai_pipeline import OCRAIPipeline, PipelineResult
from src.metadata_extraction.extractor import MetadataExtractor, DocumentMetadata
from src.metadata_extraction.storage import MetadataStorage
from src.organization_logic.engine import OrganizationEngine
from src.integration.metadata_integration import MetadataIntegrationBridge, MetadataConflict
from src.claude_integration.client import ClaudeClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_metadata_integration():
    """Demonstrate metadata integration between new AI and existing systems."""
    print("=" * 60)
    print("Metadata Integration Demo")
    print("=" * 60)
    
    # Setup components
    print("\n1. Initializing components...")
    
    # Create test data directory
    test_dir = Path("./integration_demo_temp")
    test_dir.mkdir(exist_ok=True)
    metadata_dir = test_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize existing metadata system
        metadata_storage = MetadataStorage(str(metadata_dir), use_database=True)
        organization_engine = OrganizationEngine()
        
        # Initialize Claude client (if API key available)
        claude_client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            claude_client = ClaudeClient()
            metadata_extractor = MetadataExtractor(claude_client)
        else:
            print("Note: No Anthropic API key found. Using mock mode.")
            metadata_extractor = None
        
        # Initialize OCR + AI pipeline
        ocr_processor = OCRProcessor()
        ai_summarizer = AISummarizer(claude_client=claude_client)
        pipeline = OCRAIPipeline(
            ocr_processor=ocr_processor,
            ai_summarizer=ai_summarizer,
            cache_results=False
        )
        
        # Initialize integration bridge
        integration_bridge = MetadataIntegrationBridge(
            metadata_storage=metadata_storage,
            organization_engine=organization_engine,
            conflict_resolution_strategy="confidence_based"
        )
        
        print("✓ All components initialized")
        
    except Exception as e:
        print(f"\nError initializing components: {e}")
        return
    
    # Create test documents
    print("\n2. Creating test documents...")
    
    test_documents = [
        {
            "filename": "syracuse_salt_company_1892.txt",
            "content": """
            Syracuse Salt Company
            Certificate of Stock No. 1234
            
            This certifies that John H. Smith is the owner of fifty shares
            of the capital stock of the Syracuse Salt Company, incorporated
            under the laws of the State of New York.
            
            Dated this 15th day of March, 1892
            
            Located at: Salina Street, Syracuse, New York
            
            [Signed]
            William R. Johnson, President
            Robert E. Davis, Secretary
            """,
            "existing_metadata": DocumentMetadata(
                document_type="certificate",
                categories=["legal"],
                dates=MetadataExtractor.DateInfo(document_date="1892-01-01"),
                entities=[],
                topics=["business"],
                tags=["historical"],
                summary="Stock certificate",
                suggested_folder="Legal/Certificates",
                confidence_score=0.7,
                source_file="syracuse_salt_company_1892.txt",
                processing_timestamp="2025-05-19T00:00:00"
            )
        },
        {
            "filename": "erie_canal_photo_1920.txt",
            "content": """
            [Photograph Description]
            Black and white photograph showing barges on the Erie Canal
            near Clinton Square in Syracuse, New York.
            
            Date stamped: June 1920
            
            Visible landmarks: Weighlock Building, canal boats,
            horse-drawn wagons on towpath.
            
            Written on back: "Erie Canal commerce at its peak - Syracuse"
            """,
            "existing_metadata": None  # No existing metadata
        },
        {
            "filename": "family_letter_1955.txt",
            "content": """
            Dear Margaret,
            
            I hope this letter finds you well. The weather here in Syracuse
            has been quite warm for October. The leaves at Thornden Park are
            beautiful this year.
            
            Little Johnny started school at Eastwood Elementary last month.
            He's enjoying his classes and made several new friends.
            
            Please give my regards to Uncle Frank.
            
            Love,
            Mary
            
            October 12, 1955
            """,
            "existing_metadata": DocumentMetadata(
                document_type="correspondence",
                categories=["personal"],
                dates=MetadataExtractor.DateInfo(document_date="1955-10-12"),
                entities=[
                    MetadataExtractor.Entity(name="Margaret", type="person"),
                    MetadataExtractor.Entity(name="Mary", type="person")
                ],
                topics=["family"],
                tags=["letter"],
                summary="Family correspondence",
                suggested_folder="Personal/Letters/1955",
                confidence_score=0.8,
                source_file="family_letter_1955.txt",
                processing_timestamp="2025-05-19T00:00:00"
            )
        }
    ]
    
    # Process documents
    for doc in test_documents:
        test_file = test_dir / doc["filename"]
        test_file.write_text(doc["content"])
    
    print(f"✓ Created {len(test_documents)} test documents")
    
    # Demonstrate integration for each document
    for i, doc in enumerate(test_documents):
        print(f"\n{'-' * 50}")
        print(f"3.{i+1} Processing: {doc['filename']}")
        print(f"{'-' * 50}")
        
        test_file = test_dir / doc["filename"]
        
        # Mock pipeline processing if no API key
        if not claude_client:
            print("Using mock AI results (no API key)")
            
            # Create mock results
            if "salt" in doc["content"].lower():
                ai_summary = DocumentSummary(
                    file_path=str(test_file),
                    ocr_text=doc["content"],
                    summary="Historical stock certificate from Syracuse Salt Company dated 1892",
                    category="certificate",
                    confidence_score=0.85,
                    key_entities={
                        "people": ["John H. Smith", "William R. Johnson", "Robert E. Davis"],
                        "organizations": ["Syracuse Salt Company"],
                        "locations": ["Syracuse", "Salina Street"],
                        "dates": ["1892-03-15"]
                    },
                    date_references=["1892-03-15", "1892"],
                    photo_subjects=[],
                    location_references=["Syracuse", "Salina Street"],
                    content_type="document",
                    historical_period="salt_era",
                    classification_tags=["salt_industry", "business", "certificate"],
                    claude_metadata={"model": "mock"},
                    processing_time=0.5,
                    suggested_folder_path="Hansman_Syracuse/certificate/salt_era/Syracuse_Salt_Company"
                )
            elif "canal" in doc["content"].lower():
                ai_summary = DocumentSummary(
                    file_path=str(test_file),
                    ocr_text=doc["content"],
                    summary="Historical photograph of Erie Canal at Clinton Square in Syracuse from 1920",
                    category="photo",
                    confidence_score=0.9,
                    key_entities={
                        "locations": ["Erie Canal", "Clinton Square", "Syracuse"],
                        "dates": ["1920-06"]
                    },
                    date_references=["1920-06", "1920"],
                    photo_subjects=["canal boats", "Weighlock Building", "towpath"],
                    location_references=["Erie Canal", "Clinton Square", "Syracuse"],
                    content_type="photo",
                    historical_period="canal_era",
                    classification_tags=["erie_canal", "syracuse_local", "transportation"],
                    claude_metadata={"model": "mock"},
                    processing_time=0.5,
                    suggested_folder_path="Hansman_Syracuse/photo/canal_era/Erie_Canal"
                )
            else:
                ai_summary = DocumentSummary(
                    file_path=str(test_file),
                    ocr_text=doc["content"],
                    summary="Personal family letter from Mary to Margaret discussing life in Syracuse",
                    category="letter",
                    confidence_score=0.88,
                    key_entities={
                        "people": ["Margaret", "Mary", "Johnny", "Uncle Frank"],
                        "organizations": ["Eastwood Elementary"],
                        "locations": ["Syracuse", "Thornden Park"],
                        "dates": ["1955-10-12"]
                    },
                    date_references=["1955-10-12"],
                    photo_subjects=[],
                    location_references=["Syracuse", "Thornden Park", "Eastwood"],
                    content_type="document",
                    historical_period="modern",
                    classification_tags=["personal", "family", "correspondence"],
                    claude_metadata={"model": "mock"},
                    processing_time=0.5,
                    suggested_folder_path="Hansman_Syracuse/letter/modern/Family_Correspondence"
                )
            
            pipeline_result = PipelineResult(
                file_path=test_file,
                ocr_result=OCRResult(
                    text=doc["content"],
                    confidence=0.9,
                    engine_used="mock",
                    processing_time=0.1
                ),
                ai_summary=ai_summary,
                processing_time=0.6,
                success=True
            )
        else:
            # Real processing with API
            print("Processing with Claude AI...")
            pipeline_result = pipeline.process_file(test_file)
        
        # Save existing metadata if provided
        if doc["existing_metadata"]:
            metadata_storage.save_metadata(doc["existing_metadata"])
            print("\nExisting metadata:")
            print(f"  Type: {doc['existing_metadata'].document_type}")
            print(f"  Categories: {doc['existing_metadata'].categories}")
            print(f"  Date: {doc['existing_metadata'].dates.document_date}")
            print(f"  Confidence: {doc['existing_metadata'].confidence_score}")
            print(f"  Suggested Path: {doc['existing_metadata'].suggested_folder}")
        else:
            print("\nNo existing metadata")
        
        # Display AI results
        print("\nAI-generated metadata:")
        print(f"  Type: {pipeline_result.ai_summary.content_type}")
        print(f"  Category: {pipeline_result.ai_summary.category}")
        print(f"  Period: {pipeline_result.ai_summary.historical_period}")
        print(f"  Confidence: {pipeline_result.ai_summary.confidence_score}")
        print(f"  Key Entities: {len(pipeline_result.ai_summary.key_entities)} types")
        print(f"  Suggested Path: {pipeline_result.ai_summary.suggested_folder_path}")
        
        # Integrate metadata
        print("\nIntegrating metadata...")
        integrated, conflicts = integration_bridge.integrate_pipeline_result(
            pipeline_result,
            doc["existing_metadata"],
            force_update=False
        )
        
        # Report conflicts
        if conflicts:
            print(f"\nConflicts detected ({len(conflicts)}):")
            for conflict in conflicts:
                print(f"  - {conflict.field_name}:")
                print(f"    Existing: {conflict.existing_value} (conf: {conflict.confidence_existing})")
                print(f"    New: {conflict.new_value} (conf: {conflict.confidence_new})")
                print(f"    Resolution: {conflict.resolution}")
        else:
            print("\nNo conflicts detected")
        
        # Final integrated metadata
        print("\nIntegrated metadata:")
        print(f"  Type: {integrated.document_type}")
        print(f"  Categories: {integrated.categories}")
        print(f"  Date: {integrated.dates.document_date}")
        print(f"  Entities: {len(integrated.entities)}")
        print(f"  Tags: {integrated.tags}")
        print(f"  Confidence: {integrated.confidence_score}")
        
        # Determine organization path
        target_path, rule_name = integration_bridge.update_organization_path(
            integrated,
            use_ai_suggestion=True
        )
        print(f"\nOrganization path: {target_path}")
        print(f"Rule used: {rule_name}")
        
        # Validate integration
        validation = integration_bridge.validate_integration(
            integrated,
            doc["existing_metadata"]
        )
        
        print(f"\nValidation: {'✓ Valid' if validation['is_valid'] else '✗ Invalid'}")
        if validation["errors"]:
            print(f"Errors: {validation['errors']}")
        if validation["warnings"]:
            print(f"Warnings: {validation['warnings']}")
        if validation["improvements"]:
            print(f"Improvements: {validation['improvements']}")
        
        # Save integrated metadata
        metadata_storage.save_metadata(integrated)
    
    # Create enrichment report
    print("\n" + "=" * 60)
    print("4. Creating enrichment report...")
    
    # Mock batch results for report
    all_results = []
    for doc in test_documents:
        test_file = test_dir / doc["filename"]
        
        # Create appropriate mock result based on content
        if "salt" in doc["content"].lower():
            mock_result = create_mock_result(test_file, "certificate", "salt_era")
        elif "canal" in doc["content"].lower():
            mock_result = create_mock_result(test_file, "photo", "canal_era")
        else:
            mock_result = create_mock_result(test_file, "letter", "modern")
        
        all_results.append(mock_result)
    
    enrichment_report = integration_bridge.create_enriched_metadata(
        all_results,
        batch_name="Syracuse_Historical_Demo"
    )
    
    print("\nEnrichment Report Summary:")
    print(f"  Total documents: {enrichment_report['total_documents']}")
    print(f"  OCR success: {enrichment_report['enrichment_statistics']['ocr_success']}")
    print(f"  AI classification success: {enrichment_report['enrichment_statistics']['ai_classification_success']}")
    print(f"  New entities discovered: {enrichment_report['enrichment_statistics']['new_entities_discovered']}")
    print(f"  Date references found: {enrichment_report['enrichment_statistics']['date_references_found']}")
    print(f"  Location references found: {enrichment_report['enrichment_statistics']['location_references_found']}")
    
    print("\nDocument distribution:")
    for category, count in enrichment_report["document_distribution"].items():
        print(f"  {category}: {count}")
    
    print("\nSyracuse-specific insights:")
    print(f"  Historical periods: {enrichment_report['syracuse_specific']['historical_periods']}")
    print(f"  Locations: {list(enrichment_report['syracuse_specific']['locations'].keys())}")
    print(f"  Key themes: {enrichment_report['syracuse_specific']['related_themes']}")
    
    # Export audit trail
    print("\n5. Exporting audit trail...")
    audit_path = test_dir / "audit_trail.json"
    integration_bridge.export_audit_trail(audit_path)
    print(f"✓ Audit trail exported to: {audit_path}")
    
    # Show audit events
    audit_events = integration_bridge.get_audit_trail()
    print(f"\nAudit events recorded: {len(audit_events)}")
    for event in audit_events[-3:]:  # Show last 3 events
        print(f"  - {event['event_type']} at {event['timestamp']}")
    
    # Save enrichment report
    report_path = test_dir / "enrichment_report.json"
    with open(report_path, 'w') as f:
        json.dump(enrichment_report, f, indent=2)
    print(f"\nEnrichment report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print(f"\nAll demo files saved in: {test_dir}")


def create_mock_result(file_path: Path, category: str, period: str) -> PipelineResult:
    """Create a mock pipeline result for demo purposes."""
    content = file_path.read_text()
    
    # Build entities based on content
    entities = {}
    if "John" in content or "Mary" in content:
        entities["people"] = ["John H. Smith", "Mary", "Margaret"]
    if "Syracuse" in content:
        entities["locations"] = ["Syracuse", "New York"]
    if "Company" in content:
        entities["organizations"] = ["Syracuse Salt Company"]
    
    summary = DocumentSummary(
        file_path=str(file_path),
        ocr_text=content,
        summary=f"Mock summary for {category} from {period}",
        category=category,
        confidence_score=0.85,
        key_entities=entities,
        date_references=["1892-03-15"] if "1892" in content else ["1955-10-12"],
        photo_subjects=["Erie Canal"] if category == "photo" else [],
        location_references=["Syracuse"],
        content_type="photo" if category == "photo" else "document",
        historical_period=period,
        classification_tags=[category, period],
        claude_metadata={"model": "mock"},
        processing_time=0.5,
        suggested_folder_path=f"Hansman_Syracuse/{category}/{period}"
    )
    
    return PipelineResult(
        file_path=file_path,
        ocr_result=OCRResult(
            text=content,
            confidence=0.9,
            engine_used="mock",
            processing_time=0.1
        ),
        ai_summary=summary,
        processing_time=0.6,
        success=True
    )


def cleanup_demo():
    """Clean up demo files."""
    import shutil
    demo_dir = Path("./integration_demo_temp")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
        print("\nDemo files cleaned up.")


if __name__ == "__main__":
    try:
        demo_metadata_integration()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        logger.exception("Demo failed with error")
    finally:
        # Optional cleanup
        response = input("\nClean up demo files? (y/n): ")
        if response.lower() == 'y':
            cleanup_demo()