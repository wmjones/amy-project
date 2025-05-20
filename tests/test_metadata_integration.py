"""
Test suite for metadata integration between OCR+AI pipeline and existing systems.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
from datetime import datetime
from typing import List

from src.integration.metadata_integration import (
    MetadataIntegrationBridge,
    MetadataConflict,
)
from src.metadata_extraction.extractor import DocumentMetadata, DateInfo, Entity
from src.metadata_extraction.storage import MetadataStorage
from src.organization_logic.engine import OrganizationEngine
from src.metadata_extraction.ocr_ai_pipeline import PipelineResult
from src.metadata_extraction.ai_summarizer import DocumentSummary
from src.file_access.ocr_processor import OCRResult


class TestMetadataIntegrationBridge(unittest.TestCase):
    """Test the metadata integration bridge."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_storage = Mock(spec=MetadataStorage)
        self.mock_engine = Mock(spec=OrganizationEngine)

        # Create bridge
        self.bridge = MetadataIntegrationBridge(
            metadata_storage=self.mock_storage,
            organization_engine=self.mock_engine,
            conflict_resolution_strategy="confidence_based",
        )

        # Create test data
        self.test_file = Path("test_document.jpg")
        self.create_test_data()

    def create_test_data(self):
        """Create test data for integration tests."""
        # Create existing metadata
        self.existing_metadata = DocumentMetadata(
            document_type="invoice",
            categories=["financial"],
            dates=DateInfo(document_date="2023-01-15"),
            entities=[
                Entity(name="Company A", type="organization", confidence=0.8),
                Entity(name="John Doe", type="person", confidence=0.7),
            ],
            topics=["payment", "services"],
            tags=["urgent"],
            summary="Invoice for services rendered",
            suggested_folder="Financial/Invoices/2023",
            confidence_score=0.75,
            source_file=str(self.test_file),
            processing_timestamp="2025-05-18T12:00:00",
            language="en",
        )

        # Create AI summary
        self.ai_summary = DocumentSummary(
            file_path=str(self.test_file),
            ocr_text="Invoice text content",
            summary="Detailed invoice from Company A to Company B for consulting services",
            category="business_record",
            confidence_score=0.85,
            key_entities={
                "organizations": ["Company A", "Company B"],
                "people": ["Jane Smith"],
                "dates": ["2023-01-15", "2023-02-15"],
            },
            date_references=["2023-01-15", "2023-02-15"],
            photo_subjects=[],
            location_references=["Syracuse"],
            content_type="document",
            historical_period="modern",
            classification_tags=["invoice", "financial", "business"],
            claude_metadata={"model": "claude-3", "tokens": 100},
            processing_time=2.5,
            suggested_folder_path="Hansman_Syracuse/business_record/modern/Company_A",
        )

        # Create pipeline result
        self.pipeline_result = PipelineResult(
            file_path=self.test_file,
            ocr_result=OCRResult(
                text="Invoice text content",
                confidence=0.9,
                engine_used="tesseract",
                processing_time=1.0,
            ),
            ai_summary=self.ai_summary,
            processing_time=3.5,
            success=True,
        )

    def test_initialization(self):
        """Test bridge initialization."""
        self.assertIsNotNone(self.bridge.metadata_storage)
        self.assertIsNotNone(self.bridge.organization_engine)
        self.assertEqual(self.bridge.conflict_resolution_strategy, "confidence_based")
        self.assertTrue(hasattr(self.bridge, "category_mappings"))
        self.assertTrue(hasattr(self.bridge, "entity_mappings"))

    def test_category_mappings(self):
        """Test category mapping initialization."""
        self.assertIn("photo", self.bridge.category_mappings)
        self.assertIn("personal", self.bridge.category_mappings["photo"])

        # Test reverse mappings
        self.assertIn("financial", self.bridge.reverse_category_mappings)
        self.assertIn("invoice", self.bridge.reverse_category_mappings["financial"])

    def test_integrate_pipeline_result_no_existing(self):
        """Test integration when no existing metadata exists."""
        integrated, conflicts = self.bridge.integrate_pipeline_result(
            self.pipeline_result, existing_metadata=None
        )

        # Should return AI metadata with no conflicts
        self.assertEqual(integrated.document_type, "text")  # Mapped from "document"
        self.assertEqual(integrated.categories, ["financial", "work"])  # Mapped
        self.assertEqual(len(conflicts), 0)

        # Check audit trail
        self.assertEqual(len(self.bridge.audit_trail), 1)
        self.assertEqual(self.bridge.audit_trail[0]["event_type"], "metadata_created")

    def test_integrate_pipeline_result_with_existing(self):
        """Test integration with existing metadata."""
        integrated, conflicts = self.bridge.integrate_pipeline_result(
            self.pipeline_result, existing_metadata=self.existing_metadata
        )

        # Should merge metadata
        self.assertEqual(integrated.document_type, "invoice")  # Keep existing
        self.assertIn("financial", integrated.categories)
        self.assertIn("work", integrated.categories)  # Added from AI

        # Check entities were merged
        entity_names = [e.name for e in integrated.entities]
        self.assertIn("Company A", entity_names)
        self.assertIn("Company B", entity_names)  # Added from AI
        self.assertIn("Jane Smith", entity_names)  # Added from AI

        # Should have higher confidence
        self.assertGreaterEqual(
            integrated.confidence_score, self.existing_metadata.confidence_score
        )

    def test_conflict_detection(self):
        """Test conflict detection between metadata sources."""
        # Modify AI summary to create conflicts
        self.ai_summary.date_references = ["2023-02-20"]  # Different date

        integrated, conflicts = self.bridge.integrate_pipeline_result(
            self.pipeline_result, existing_metadata=self.existing_metadata
        )

        # Should detect date conflict
        self.assertGreater(len(conflicts), 0)

        date_conflict = next(
            (c for c in conflicts if c.field_name == "document_date"), None
        )
        self.assertIsNotNone(date_conflict)
        self.assertEqual(date_conflict.existing_value, "2023-01-15")
        self.assertEqual(date_conflict.new_value, "2023-02-20")

    def test_conflict_resolution_confidence_based(self):
        """Test confidence-based conflict resolution."""
        # Create conflict with lower confidence AI result
        self.ai_summary.confidence_score = 0.6  # Lower than existing
        self.ai_summary.date_references = ["2023-02-20"]

        integrated, conflicts = self.bridge.integrate_pipeline_result(
            self.pipeline_result, existing_metadata=self.existing_metadata
        )

        # Should keep existing date due to higher confidence
        self.assertEqual(integrated.dates.document_date, "2023-01-15")

        # Check conflict resolution
        date_conflict = next(
            (c for c in conflicts if c.field_name == "document_date"), None
        )
        self.assertEqual(date_conflict.resolution, "kept_existing")

    def test_conflict_resolution_ai_preferred(self):
        """Test AI-preferred conflict resolution."""
        self.bridge.conflict_resolution_strategy = "ai_preferred"

        self.ai_summary.date_references = ["2023-02-20"]

        integrated, conflicts = self.bridge.integrate_pipeline_result(
            self.pipeline_result, existing_metadata=self.existing_metadata
        )

        # Should use AI date regardless of confidence
        self.assertEqual(integrated.dates.document_date, "2023-02-20")

    def test_convert_ai_to_document_metadata(self):
        """Test conversion from AI format to DocumentMetadata."""
        doc_metadata = self.bridge._convert_ai_to_document_metadata(
            self.pipeline_result
        )

        self.assertIsInstance(doc_metadata, DocumentMetadata)
        self.assertEqual(doc_metadata.document_type, "text")  # Mapped from "document"
        self.assertIn("financial", doc_metadata.categories)  # Mapped category
        self.assertEqual(len(doc_metadata.entities), 4)  # All entities converted

        # Check entity mapping
        org_entities = [e for e in doc_metadata.entities if e.type == "organization"]
        self.assertEqual(len(org_entities), 2)

    def test_merge_metadata(self):
        """Test metadata merging logic."""
        ai_metadata = self.bridge._convert_ai_to_document_metadata(self.pipeline_result)

        merged, conflicts = self.bridge._merge_metadata(
            self.existing_metadata, ai_metadata, self.ai_summary.confidence_score
        )

        # Check merged categories
        self.assertIn("financial", merged.categories)
        self.assertIn("work", merged.categories)

        # Check merged entities
        self.assertGreater(len(merged.entities), len(self.existing_metadata.entities))

        # Check merged topics and tags
        self.assertIn("invoice", merged.tags)
        self.assertIn("business", merged.tags)

    def test_update_organization_path(self):
        """Test organization path update with AI suggestions."""
        # Mock organization engine response
        self.mock_engine.determine_target_location.return_value = (
            "Financial/Invoices/2023",
            "Financial Documents by Year",
        )

        # Test with high confidence AI suggestion
        metadata = self.existing_metadata
        metadata.suggested_folder = "Hansman_Syracuse/business/2023/Company_A"
        metadata.confidence_score = 0.9

        target_path, rule_name = self.bridge.update_organization_path(
            metadata, use_ai_suggestion=True
        )

        # Should use AI suggestion due to high confidence and more specific path
        self.assertEqual(target_path, "Hansman_Syracuse/business/2023/Company_A")
        self.assertEqual(rule_name, "AI Enhanced Organization")

    def test_create_enriched_metadata(self):
        """Test enriched metadata report creation."""
        results = [self.pipeline_result]

        report = self.bridge.create_enriched_metadata(results, batch_name="Test_Batch")

        self.assertEqual(report["batch_name"], "Test_Batch")
        self.assertEqual(report["total_documents"], 1)
        self.assertEqual(report["enrichment_statistics"]["ocr_success"], 1)
        self.assertEqual(
            report["enrichment_statistics"]["ai_classification_success"], 1
        )

        # Check Syracuse-specific data
        self.assertIn("modern", report["syracuse_specific"]["historical_periods"])
        self.assertIn("Syracuse", report["syracuse_specific"]["locations"])

    def test_validate_integration(self):
        """Test metadata validation."""
        # Test valid metadata
        validation = self.bridge.validate_integration(self.existing_metadata)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(len(validation["errors"]), 0)

        # Test invalid metadata
        invalid_metadata = DocumentMetadata(
            document_type="",  # Missing type
            categories=[],  # Missing categories
            dates=DateInfo(),
            entities=[],
            topics=[],
            tags=[],
            summary="",
            suggested_folder="invalid<>path",  # Invalid characters
            confidence_score=1.5,  # Out of range
            source_file="test.txt",
            processing_timestamp=datetime.now().isoformat(),
        )

        validation = self.bridge.validate_integration(invalid_metadata)

        self.assertFalse(validation["is_valid"])
        self.assertGreater(len(validation["errors"]), 0)
        self.assertGreater(len(validation["warnings"]), 0)

    def test_validation_improvements(self):
        """Test validation improvement detection."""
        # Create improved metadata
        improved_metadata = DocumentMetadata(
            document_type=self.existing_metadata.document_type,
            categories=self.existing_metadata.categories,
            dates=self.existing_metadata.dates,
            entities=self.existing_metadata.entities
            + [Entity(name="New Entity", type="organization")],
            topics=self.existing_metadata.topics,
            tags=self.existing_metadata.tags,
            summary="Enhanced summary with more details",
            suggested_folder=self.existing_metadata.suggested_folder,
            confidence_score=0.9,  # Higher confidence
            source_file=self.existing_metadata.source_file,
            processing_timestamp=datetime.now().isoformat(),
        )

        validation = self.bridge.validate_integration(
            improved_metadata, self.existing_metadata
        )

        self.assertTrue(validation["is_valid"])
        self.assertGreater(len(validation["improvements"]), 0)

        # Check specific improvements
        improvements_text = " ".join(validation["improvements"])
        self.assertIn("entities", improvements_text)
        self.assertIn("confidence", improvements_text)

    def test_audit_trail(self):
        """Test audit trail functionality."""
        # Perform some operations
        self.bridge.integrate_pipeline_result(self.pipeline_result)

        # Check audit trail
        audit_events = self.bridge.get_audit_trail()
        self.assertGreater(len(audit_events), 0)

        # Test filtering
        filtered_events = self.bridge.get_audit_trail(event_type="metadata_created")
        self.assertTrue(
            all(e["event_type"] == "metadata_created" for e in filtered_events)
        )

    def test_export_audit_trail(self):
        """Test audit trail export."""
        # Add some events
        self.bridge._log_audit_event("test_event", "test_file.txt", {"test": "data"})

        # Mock file operations
        mock_open = unittest.mock.mock_open()
        with patch("builtins.open", mock_open):
            self.bridge.export_audit_trail(Path("audit.json"))

        # Check that file was written
        mock_open.assert_called_once_with(Path("audit.json"), "w")

        # Check JSON was written
        written_data = ""
        for call in mock_open().write.call_args_list:
            written_data += call[0][0]

        audit_data = json.loads(written_data)
        self.assertIsInstance(audit_data, list)
        self.assertGreater(len(audit_data), 0)

    def test_date_normalization(self):
        """Test date normalization functionality."""
        # Test ISO format (no change)
        iso_date = self.bridge._normalize_date("2023-01-15")
        self.assertEqual(iso_date, "2023-01-15")

        # Test year extraction
        year_only = self.bridge._normalize_date("1892")
        self.assertEqual(year_only, "1892-01-01")

        # Test invalid date
        invalid = self.bridge._normalize_date("invalid date")
        self.assertIsNone(invalid)

    def test_entity_mapping(self):
        """Test entity type mapping."""
        # Create AI summary with different entity types
        ai_summary = DocumentSummary(
            file_path=str(self.test_file),
            ocr_text="Test",
            summary="Test",
            category="document",
            confidence_score=0.8,
            key_entities={
                "people": ["John Doe"],
                "organizations": ["Company X"],
                "locations": ["Syracuse"],
                "dates": ["2023-01-01"],
            },
            date_references=[],
            photo_subjects=[],
            location_references=[],
            content_type="document",
            historical_period="modern",
            classification_tags=[],
            claude_metadata={},
            processing_time=1.0,
        )

        pipeline_result = PipelineResult(
            file_path=self.test_file,
            ocr_result=Mock(),
            ai_summary=ai_summary,
            processing_time=1.0,
            success=True,
        )

        doc_metadata = self.bridge._convert_ai_to_document_metadata(pipeline_result)

        # Check entity mapping
        entity_types = {e.type for e in doc_metadata.entities}
        self.assertIn("person", entity_types)  # Mapped from "people"
        self.assertIn("organization", entity_types)  # Mapped from "organizations"
        self.assertIn("location", entity_types)  # Mapped from "locations"


if __name__ == "__main__":
    unittest.main()
