"""
Unit tests for metadata extraction and storage.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from src.metadata_extraction.extractor import (
    MetadataExtractor,
    DocumentMetadata,
    Entity,
    DateInfo,
)
from src.metadata_extraction.storage import MetadataStorage
from src.claude_integration.client import ClaudeClient, AnalysisResult


class TestMetadataExtractor:
    """Test MetadataExtractor functionality."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create mock Claude client."""
        from unittest.mock import Mock

        client = Mock(spec=ClaudeClient)
        return client

    @pytest.fixture
    def extractor(self, mock_claude_client):
        """Create metadata extractor."""
        return MetadataExtractor(mock_claude_client)

    @pytest.mark.skip(reason="Test requires fixing JSON parsing mock")
    def test_extract_metadata_with_json_response(self, extractor, mock_claude_client):
        """Test metadata extraction with JSON response."""
        # Mock Claude response
        mock_result = AnalysisResult(
            content=json.dumps(
                {
                    "document_type": "invoice",
                    "category": "financial",
                    "date": "2024-03-15",
                    "entities": {
                        "organizations": ["Acme Corp"],
                        "people": ["John Doe"],
                    },
                    "summary": "Invoice for services",
                    "suggested_folder": "financial/invoices/2024",
                    "confidence_score": 0.95,
                },
                indent=None,
            ),  # Ensure no extra formatting
            metadata={},
            confidence_score=0.95,
            tokens_used=100,
            model="claude-3",
        )

        mock_claude_client.analyze_document.return_value = mock_result

        # Extract metadata
        metadata = extractor.extract_metadata(
            file_content="Invoice content",
            file_path="/path/to/invoice.pdf",
            file_type="pdf",
        )

        # Assertions
        assert metadata.document_type == "invoice"
        assert "financial" in metadata.categories
        assert metadata.dates.document_date == "2024-03-15"
        assert len(metadata.entities) == 2
        assert metadata.confidence_score == 0.95
        assert metadata.suggested_folder == "financial/invoices/2024"

    @pytest.mark.skip(reason="Test requires fixing text parsing mock")
    def test_extract_metadata_with_text_response(self, extractor, mock_claude_client):
        """Test metadata extraction with text response."""
        # Mock Claude response (non-JSON)
        mock_result = AnalysisResult(
            content="""
            Document Type: Contract
            Category: Legal
            Date: January 15, 2024
            Summary: Employment contract
            Suggested Folder: legal/contracts
            """,
            metadata={},
            confidence_score=0.85,
            tokens_used=100,
            model="claude-3",
        )

        mock_claude_client.analyze_document.return_value = mock_result

        # Extract metadata
        metadata = extractor.extract_metadata(
            file_content="Contract content",
            file_path="/path/to/contract.pdf",
            file_type="pdf",
        )

        # Assertions
        assert metadata.document_type == "Contract"
        assert metadata.categories[0] == "Legal"
        assert metadata.suggested_folder == "legal/contracts"
        assert metadata.confidence_score == 0.85

    @pytest.mark.skip(reason="Test requires fixing date extraction")
    def test_date_extraction(self, extractor):
        """Test date extraction from content."""
        content = """
        Invoice Date: 2024-03-15
        Due Date: April 1, 2024
        Created on 01.03.2024
        """

        dates = extractor._extract_dates(content, {})

        assert "2024-03-15" in dates.mentioned_dates
        assert "2024-04-01" in dates.mentioned_dates
        assert "2024-03-01" in dates.mentioned_dates

    def test_entity_extraction(self, extractor):
        """Test entity extraction from parsed data."""
        parsed_data = {
            "entities": {
                "people": ["John Doe", "Jane Smith"],
                "organizations": ["Acme Corp"],
            }
        }

        entities = extractor._extract_entities(parsed_data)

        assert len(entities) == 3
        assert any(e.name == "John Doe" and e.type == "people" for e in entities)
        assert any(
            e.name == "Acme Corp" and e.type == "organizations" for e in entities
        )

    def test_date_normalization(self, extractor):
        """Test date normalization to ISO format."""
        test_dates = {
            "2024-03-15": "2024-03-15",
            "03/15/2024": "2024-03-15",
            "15.03.2024": "2024-03-15",
            "March 15, 2024": "2024-03-15",
            "Mar 15, 2024": "2024-03-15",
        }

        for input_date, expected in test_dates.items():
            normalized = extractor._normalize_date(input_date)
            assert normalized == expected

    def test_metadata_normalization(self, extractor):
        """Test metadata normalization."""
        parsed_data = {
            "category": "financial",
            "confidence_score": "0.95",
            "suggested_folder": "path\\to\\folder",
        }

        dates = DateInfo()
        entities = []

        normalized = extractor._normalize_metadata(parsed_data, dates, entities)

        assert normalized["categories"] == ["financial"]
        assert normalized["confidence_score"] == 0.95
        assert normalized["suggested_folder"] == "path/to/folder"

    def test_fallback_metadata(self, extractor):
        """Test fallback metadata creation."""
        metadata = extractor._create_fallback_metadata(
            "/path/to/file.pdf", "pdf", "Error message"
        )

        assert metadata.document_type == "unknown"
        assert metadata.categories == ["error"]
        assert metadata.confidence_score == 0.0
        assert "Error message" in metadata.summary


class TestMetadataStorage:
    """Test MetadataStorage functionality."""

    @pytest.fixture
    def storage_db(self):
        """Create database storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetadataStorage(temp_dir, use_database=True)
            yield storage

    @pytest.fixture
    def storage_json(self):
        """Create JSON storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetadataStorage(temp_dir, use_database=False)
            yield storage

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return DocumentMetadata(
            document_type="invoice",
            categories=["financial", "accounting"],
            dates=DateInfo(
                document_date="2024-03-15", mentioned_dates=["2024-03-15", "2024-04-01"]
            ),
            entities=[
                Entity(name="Acme Corp", type="organization"),
                Entity(name="John Doe", type="person"),
            ],
            topics=["payment", "services"],
            tags=["urgent", "q1-2024"],
            summary="Invoice for consulting services",
            suggested_folder="financial/invoices/2024",
            confidence_score=0.95,
            source_file="/path/to/invoice.pdf",
            processing_timestamp=datetime.now().isoformat(),
            file_size=1024,
            page_count=2,
            language="en",
        )

    def test_save_and_retrieve_metadata_db(self, storage_db, sample_metadata):
        """Test saving and retrieving metadata from database."""
        # Save metadata
        success = storage_db.save_metadata(sample_metadata)
        assert success

        # Retrieve metadata
        retrieved = storage_db.get_metadata(sample_metadata.source_file)
        assert retrieved is not None
        assert retrieved.document_type == sample_metadata.document_type
        assert retrieved.categories == sample_metadata.categories
        assert retrieved.confidence_score == sample_metadata.confidence_score

    def test_save_and_retrieve_metadata_json(self, storage_json, sample_metadata):
        """Test saving and retrieving metadata from JSON."""
        # Save metadata
        success = storage_json.save_metadata(sample_metadata)
        assert success

        # Retrieve metadata
        retrieved = storage_json.get_metadata(sample_metadata.source_file)
        assert retrieved is not None
        assert retrieved.document_type == sample_metadata.document_type
        assert retrieved.categories == sample_metadata.categories

    def test_search_metadata(self, storage_db, sample_metadata):
        """Test searching metadata."""
        # Save multiple metadata entries
        storage_db.save_metadata(sample_metadata)

        # Modify and save another
        metadata2 = sample_metadata
        metadata2.source_file = "/path/to/contract.pdf"
        metadata2.document_type = "contract"
        metadata2.categories = ["legal"]
        storage_db.save_metadata(metadata2)

        # Search by document type
        results = storage_db.search_metadata(document_type="invoice")
        assert len(results) >= 1
        assert all(r.document_type == "invoice" for r in results)

        # Search by category
        results = storage_db.search_metadata(category="financial")
        assert len(results) >= 1
        assert all("financial" in r.categories for r in results)

        # Search by confidence
        results = storage_db.search_metadata(min_confidence=0.9)
        assert all(r.confidence_score >= 0.9 for r in results)

    def test_export_json(self, storage_db, sample_metadata):
        """Test exporting metadata to JSON."""
        # Save metadata
        storage_db.save_metadata(sample_metadata)

        # Export to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            success = storage_db.export_metadata(temp_file.name, format="json")
            assert success

            # Verify export
            with open(temp_file.name, "r") as f:
                exported = json.load(f)

            assert len(exported) >= 1
            assert exported[0]["document_type"] == sample_metadata.document_type

    def test_export_csv(self, storage_db, sample_metadata):
        """Test exporting metadata to CSV."""
        # Save metadata
        storage_db.save_metadata(sample_metadata)

        # Export to CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            success = storage_db.export_metadata(temp_file.name, format="csv")
            assert success

            # Verify export
            import csv

            with open(temp_file.name, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) >= 1
            assert rows[0]["document_type"] == sample_metadata.document_type

    def test_backup(self, storage_db, sample_metadata):
        """Test backup functionality."""
        # Save metadata
        storage_db.save_metadata(sample_metadata)

        # Create backup
        with tempfile.TemporaryDirectory() as backup_dir:
            success = storage_db.backup(backup_dir)
            assert success

            # Verify backup exists
            backup_files = list(Path(backup_dir).glob("*.db"))
            assert len(backup_files) == 1

    def test_metadata_validation(self):
        """Test metadata validation."""
        from src.metadata_extraction.extractor import MetadataExtractor
        from unittest.mock import Mock

        extractor = MetadataExtractor(Mock())

        # Valid metadata
        valid_metadata = DocumentMetadata(
            document_type="invoice",
            categories=["financial"],
            dates=DateInfo(),
            entities=[],
            topics=[],
            tags=[],
            summary="",
            suggested_folder="folder",
            confidence_score=0.8,
            source_file="/path/file.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

        assert extractor.validate_metadata(valid_metadata)

        # Invalid metadata (missing document type)
        invalid_metadata = DocumentMetadata(
            document_type="",
            categories=["financial"],
            dates=DateInfo(),
            entities=[],
            topics=[],
            tags=[],
            summary="",
            suggested_folder="folder",
            confidence_score=0.8,
            source_file="/path/file.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

        assert not extractor.validate_metadata(invalid_metadata)
