#!/usr/bin/env python3
"""
Demo script showing metadata extraction and storage.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from src.metadata_extraction.extractor import (
    MetadataExtractor,
    DocumentMetadata,
    DateInfo,
    Entity,
)
from src.metadata_extraction.storage import MetadataStorage
from src.claude_integration.client import ClaudeClient, AnalysisResult

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_metadata_extraction():
    """Demonstrate metadata extraction from documents."""
    print("=== Metadata Extraction Demo ===\n")

    # Mock Claude client for demo (in real usage, use actual client)
    from unittest.mock import Mock

    mock_claude = Mock(spec=ClaudeClient)

    # Create extractor
    extractor = MetadataExtractor(mock_claude)

    # Example 1: Invoice
    print("Example 1: Processing Invoice")
    mock_claude.analyze_document.return_value = AnalysisResult(
        content=json.dumps(
            {
                "document_type": "invoice",
                "category": "financial",
                "date": "2024-03-15",
                "entities": {
                    "organizations": ["ABC Company", "XYZ Corp"],
                    "people": ["John Smith"],
                },
                "topics": ["professional services", "consulting"],
                "summary": "Invoice for Q1 consulting services totaling $15,000",
                "suggested_folder": "financial/invoices/2024/Q1",
                "confidence_score": 0.95,
                "tags": ["paid", "Q1-2024", "consulting"],
            }
        ),
        metadata={},
        confidence_score=0.95,
        tokens_used=150,
        model="claude-3",
    )

    invoice_metadata = extractor.extract_metadata(
        file_content="Invoice #12345...",
        file_path="/docs/invoice_12345.pdf",
        file_type="pdf",
        file_size=102400,
    )

    print(f"Document Type: {invoice_metadata.document_type}")
    print(f"Categories: {invoice_metadata.categories}")
    print(f"Date: {invoice_metadata.dates.document_date}")
    print(f"Entities: {[f'{e.name} ({e.type})' for e in invoice_metadata.entities]}")
    print(f"Suggested Folder: {invoice_metadata.suggested_folder}")
    print(f"Confidence: {invoice_metadata.confidence_score}")
    print()

    # Example 2: Contract
    print("Example 2: Processing Contract")
    mock_claude.analyze_document.return_value = AnalysisResult(
        content="""
        Document Type: Employment Contract
        Category: Legal
        Date: January 1, 2024
        Parties: John Doe, Acme Corporation
        Summary: Standard employment agreement with 2-year term
        Suggested Folder: legal/contracts/employment/2024
        """,
        metadata={},
        confidence_score=0.87,
        tokens_used=120,
        model="claude-3",
    )

    contract_metadata = extractor.extract_metadata(
        file_content="This employment agreement...",
        file_path="/docs/employment_contract.pdf",
        file_type="pdf",
    )

    print(f"Document Type: {contract_metadata.document_type}")
    print(f"Categories: {contract_metadata.categories}")
    print(f"Summary: {contract_metadata.summary}")
    print()


def demo_metadata_storage():
    """Demonstrate metadata storage and retrieval."""
    print("\n=== Metadata Storage Demo ===\n")

    # Create temporary storage
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize storage (using database)
        storage = MetadataStorage(temp_dir, use_database=True)

        # Create sample metadata
        metadata1 = DocumentMetadata(
            document_type="invoice",
            categories=["financial", "accounting"],
            dates=DateInfo(
                document_date="2024-03-15", mentioned_dates=["2024-03-15", "2024-04-15"]
            ),
            entities=[
                Entity(name="ABC Corp", type="organization"),
                Entity(name="John Doe", type="person"),
            ],
            topics=["payment", "services"],
            tags=["urgent", "Q1-2024"],
            summary="Invoice for March services",
            suggested_folder="financial/invoices/2024/03",
            confidence_score=0.92,
            source_file="/docs/invoice_001.pdf",
            processing_timestamp=datetime.now().isoformat(),
            file_size=51200,
            language="en",
        )

        metadata2 = DocumentMetadata(
            document_type="report",
            categories=["business", "quarterly"],
            dates=DateInfo(document_date="2024-03-31"),
            entities=[Entity(name="Marketing Dept", type="organization")],
            topics=["sales", "performance", "Q1 results"],
            tags=["quarterly", "2024"],
            summary="Q1 2024 performance report",
            suggested_folder="reports/quarterly/2024",
            confidence_score=0.88,
            source_file="/docs/q1_report.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

        # Save metadata
        print("Saving metadata...")
        storage.save_metadata(metadata1)
        storage.save_metadata(metadata2)
        print("✅ Metadata saved")
        print()

        # Retrieve metadata
        print("Retrieving metadata...")
        retrieved = storage.get_metadata("/docs/invoice_001.pdf")
        if retrieved:
            print(f"Retrieved: {retrieved.document_type} - {retrieved.summary}")
        print()

        # Search metadata
        print("Searching for financial documents...")
        results = storage.search_metadata(category="financial")
        print(f"Found {len(results)} financial documents")
        for result in results:
            print(f"  - {result.source_file}: {result.summary}")
        print()

        # Search by date
        print("Searching for documents after 2024-03-01...")
        results = storage.search_metadata(date_after="2024-03-01")
        print(f"Found {len(results)} documents")
        for result in results:
            print(f"  - {result.source_file}: {result.dates.document_date}")
        print()

        # Export metadata
        print("Exporting metadata...")
        export_path_json = Path(temp_dir) / "metadata_export.json"
        export_path_csv = Path(temp_dir) / "metadata_export.csv"

        storage.export_metadata(str(export_path_json), format="json")
        storage.export_metadata(str(export_path_csv), format="csv")

        print(f"✅ Exported to {export_path_json}")
        print(f"✅ Exported to {export_path_csv}")

        # Show JSON export
        with open(export_path_json, "r") as f:
            exported_data = json.load(f)

        print(f"\nExported {len(exported_data)} metadata entries")
        print(json.dumps(exported_data[0], indent=2)[:500] + "...")


def demo_date_extraction():
    """Demonstrate date extraction capabilities."""
    print("\n=== Date Extraction Demo ===\n")

    from unittest.mock import Mock

    extractor = MetadataExtractor(Mock())

    test_content = """
    Invoice Date: March 15, 2024
    Due Date: 04/15/2024
    Contract signed on 01.03.2024
    Expires: 2025-03-01
    Meeting scheduled for Jan 10, 2024
    """

    dates = extractor._extract_dates(test_content, {})

    print("Extracted dates:")
    print(f"Document date: {dates.document_date}")
    print(f"Mentioned dates: {dates.mentioned_dates}")

    # Test date normalization
    print("\nDate normalization examples:")
    test_dates = [
        "2024-03-15",
        "03/15/2024",
        "15.03.2024",
        "March 15, 2024",
        "Mar 15, 2024",
    ]

    for date_str in test_dates:
        normalized = extractor._normalize_date(date_str)
        print(f"{date_str:20} -> {normalized}")


def main():
    """Run all demos."""
    print("Metadata System Demo")
    print("===================")

    demo_metadata_extraction()
    print("\n" + "=" * 50 + "\n")

    demo_metadata_storage()
    print("\n" + "=" * 50 + "\n")

    demo_date_extraction()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
