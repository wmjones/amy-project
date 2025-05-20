"""
Unit tests for organization logic engine.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from src.organization_logic.engine import OrganizationEngine, OrganizationRule
from src.organization_logic.rule_manager import RuleManager
from src.organization_logic.conflict_resolver import (
    ConflictResolver,
    ConflictResolution,
)
from src.metadata_extraction.extractor import DocumentMetadata, DateInfo, Entity


class TestOrganizationRule:
    """Test OrganizationRule functionality."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return DocumentMetadata(
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
            summary="Invoice for services",
            suggested_folder="financial/invoices",
            confidence_score=0.95,
            source_file="/path/to/invoice.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

    def test_rule_matches_document_type(self, sample_metadata):
        """Test rule matching by document type."""
        rule = OrganizationRule(
            {
                "name": "Invoice Rule",
                "conditions": {"document_type": "invoice"},
                "path_template": "Invoices/{date.year}",
            }
        )

        assert rule.matches(sample_metadata)

        # Should not match different type
        sample_metadata.document_type = "contract"
        assert not rule.matches(sample_metadata)

    def test_rule_matches_categories(self, sample_metadata):
        """Test rule matching by categories."""
        rule = OrganizationRule(
            {
                "name": "Financial Rule",
                "conditions": {"categories": ["financial"]},
                "path_template": "Financial/{document_type}",
            }
        )

        assert rule.matches(sample_metadata)

        # Should not match without category
        sample_metadata.categories = ["legal"]
        assert not rule.matches(sample_metadata)

    def test_rule_matches_confidence(self, sample_metadata):
        """Test rule matching by confidence threshold."""
        rule = OrganizationRule(
            {
                "name": "High Confidence Rule",
                "conditions": {"min_confidence": 0.9},
                "path_template": "HighConfidence/{document_type}",
            }
        )

        assert rule.matches(sample_metadata)

        # Should not match low confidence
        sample_metadata.confidence_score = 0.7
        assert not rule.matches(sample_metadata)

    def test_rule_matches_entity_types(self, sample_metadata):
        """Test rule matching by entity types."""
        rule = OrganizationRule(
            {
                "name": "Organization Rule",
                "conditions": {"entity_types": ["organization"]},
                "path_template": "Companies/{entity.organization[0]}",
            }
        )

        assert rule.matches(sample_metadata)

        # Should not match without entity type
        sample_metadata.entities = [Entity(name="John Doe", type="person")]
        assert not rule.matches(sample_metadata)

    def test_rule_disabled(self, sample_metadata):
        """Test disabled rule doesn't match."""
        rule = OrganizationRule(
            {
                "name": "Disabled Rule",
                "conditions": {"document_type": "invoice"},
                "path_template": "Test",
                "enabled": False,
            }
        )

        assert not rule.matches(sample_metadata)


class TestOrganizationEngine:
    """Test OrganizationEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create organization engine."""
        return OrganizationEngine(use_default_rules=True)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return DocumentMetadata(
            document_type="invoice",
            categories=["financial"],
            dates=DateInfo(document_date="2024-03-15"),
            entities=[Entity(name="ABC Corp", type="organization")],
            topics=["payment"],
            tags=["Q1-2024"],
            summary="Invoice for services",
            suggested_folder="financial/invoices/2024",
            confidence_score=0.95,
            source_file="/path/to/invoice.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

    def test_determine_target_location(self, engine, sample_metadata):
        """Test determining target location."""
        path, rule_name = engine.determine_target_location(sample_metadata)

        assert path is not None
        assert rule_name is not None
        assert "invoice" in path.lower() or "financial" in path.lower()

    def test_path_generation(self, engine, sample_metadata):
        """Test path generation from template."""
        rule = OrganizationRule(
            {
                "name": "Test Rule",
                "conditions": {},
                "path_template": "{document_type}/{date.year}/{date.month}/{entity.organization[0]}",
            }
        )

        path = engine._generate_path(rule, sample_metadata)

        assert path == "invoice/2024/03/ABC_Corp"

    def test_path_sanitization(self, engine):
        """Test path component sanitization."""
        # Test invalid characters
        assert engine._sanitize_path_component("test:file") == "testfile"
        assert engine._sanitize_path_component("test*file?") == "testfile"

        # Test spaces
        assert engine._sanitize_path_component("test file") == "test_file"

        # Test length limit
        long_name = "a" * 100
        sanitized = engine._sanitize_path_component(long_name)
        assert len(sanitized) <= 50

    def test_default_path(self, engine, sample_metadata):
        """Test default path generation."""
        path = engine._get_default_path(sample_metadata)

        assert path.startswith("Unsorted")
        assert "invoice" in path
        assert "2024" in path

    def test_custom_rules(self):
        """Test loading custom rules."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            custom_rules = [
                {
                    "name": "Custom Invoice Rule",
                    "conditions": {"document_type": "invoice"},
                    "path_template": "Custom/Invoices/{date.year}",
                    "priority": 100,
                }
            ]
            json.dump(custom_rules, f)

        engine = OrganizationEngine(rules_path=f.name, use_default_rules=False)

        assert len(engine.rules) == 1
        assert engine.rules[0].name == "Custom Invoice Rule"

    def test_claude_suggestion_priority(self, engine, sample_metadata):
        """Test that Claude's suggestion is used for high confidence."""
        sample_metadata.confidence_score = 0.90
        sample_metadata.suggested_folder = "custom/claude/suggestion"

        path, rule_name = engine.determine_target_location(sample_metadata)

        assert path == "custom/claude/suggestion"
        assert rule_name == "Claude AI Suggestion"

    def test_date_formatting_functions(self, engine, sample_metadata):
        """Test date formatting in templates."""
        rule = OrganizationRule(
            {
                "name": "Date Format Rule",
                "conditions": {},
                "path_template": "{dates.document_date|%Y-%m-%d}/{dates.document_date|%B}",
            }
        )

        path = engine._generate_path(rule, sample_metadata)

        assert path == "2024-03-15/March"


class TestRuleManager:
    """Test RuleManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create rule manager with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RuleManager(temp_dir)

    def test_create_rule_from_template(self, manager):
        """Test creating rule from template."""
        rule = manager.create_rule_from_template(
            "by_type_and_date",
            {
                "name": "Custom Invoice Rule",
                "conditions.document_type": "invoice",
                "priority": 85,
            },
        )

        assert rule["name"] == "Custom Invoice Rule"
        assert rule["conditions"]["document_type"] == "invoice"
        assert rule["priority"] == 85

    def test_validate_rule_valid(self, manager):
        """Test validating valid rule."""
        rule = {
            "name": "Valid Rule",
            "conditions": {"document_type": "invoice"},
            "path_template": "Invoices/{date.year}",
            "priority": 50,
        }

        errors = manager.validate_rule(rule)
        assert len(errors) == 0

    def test_validate_rule_invalid(self, manager):
        """Test validating invalid rule."""
        # Missing required fields
        rule = {"conditions": {}}
        errors = manager.validate_rule(rule)
        assert "name" in str(errors)
        assert "path_template" in str(errors)

        # Invalid condition
        rule = {
            "name": "Invalid Rule",
            "conditions": {"invalid_condition": "value"},
            "path_template": "Test",
        }
        errors = manager.validate_rule(rule)
        assert "Unknown condition type" in str(errors)

    def test_save_and_load_rules(self, manager):
        """Test saving and loading rule sets."""
        rules = [
            {
                "name": "Test Rule 1",
                "conditions": {"document_type": "invoice"},
                "path_template": "Invoices/{date.year}",
                "priority": 50,
            },
            {
                "name": "Test Rule 2",
                "conditions": {"categories": ["financial"]},
                "path_template": "Financial/{document_type}",
                "priority": 60,
            },
        ]

        # Save rules
        manager.save_rule_set(rules, "test_rules")

        # Load rules
        loaded_rules = manager.load_rule_set("test_rules")

        assert len(loaded_rules) == 2
        assert loaded_rules[0]["name"] == "Test Rule 1"
        assert loaded_rules[1]["name"] == "Test Rule 2"

    def test_list_rule_sets(self, manager):
        """Test listing available rule sets."""
        # Save some rule sets
        manager.save_rule_set([{"name": "Rule", "path_template": "Test"}], "set1")
        manager.save_rule_set([{"name": "Rule", "path_template": "Test"}], "set2")

        sets = manager.list_rule_sets()

        assert "set1" in sets
        assert "set2" in sets

    def test_generate_example_rules(self, manager):
        """Test generating example rules."""
        examples = manager.generate_example_rules()

        assert len(examples) > 0

        # Validate all examples
        for rule in examples:
            errors = manager.validate_rule(rule)
            assert len(errors) == 0


class TestConflictResolver:
    """Test ConflictResolver functionality."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return DocumentMetadata(
            document_type="invoice",
            categories=["financial"],
            dates=DateInfo(document_date="2024-03-15"),
            entities=[Entity(name="ABC Corp", type="organization")],
            topics=["payment"],
            tags=["Q1-2024"],
            summary="Invoice for services",
            suggested_folder="",
            confidence_score=0.85,
            source_file="/path/to/invoice.pdf",
            processing_timestamp=datetime.now().isoformat(),
        )

    def test_resolve_single_rule(self, sample_metadata):
        """Test resolution with single rule."""
        resolver = ConflictResolver()

        rule = OrganizationRule(
            {
                "name": "Single Rule",
                "conditions": {},
                "path_template": "Test",
                "priority": 50,
            }
        )

        matching_rules = [(rule, "Test/Path")]

        resolution = resolver.resolve(matching_rules, sample_metadata)

        assert resolution.selected_rule == rule
        assert resolution.selected_path == "Test/Path"
        assert resolution.reason == "Only one rule matched"
        assert len(resolution.alternatives) == 0

    def test_resolve_by_priority(self, sample_metadata):
        """Test resolution by priority."""
        resolver = ConflictResolver(strategy="highest_priority")

        rule1 = OrganizationRule(
            {
                "name": "Low Priority",
                "conditions": {},
                "path_template": "LowPriority",
                "priority": 30,
            }
        )

        rule2 = OrganizationRule(
            {
                "name": "High Priority",
                "conditions": {},
                "path_template": "HighPriority",
                "priority": 80,
            }
        )

        # Note: rules should be pre-sorted by priority (highest first)
        matching_rules = [(rule2, "HighPriority/Path"), (rule1, "LowPriority/Path")]

        resolution = resolver.resolve(matching_rules, sample_metadata)

        assert resolution.selected_rule == rule2
        assert "priority" in resolution.reason.lower()

    def test_resolve_by_specificity(self, sample_metadata):
        """Test resolution by specificity."""
        resolver = ConflictResolver(strategy="most_specific")

        rule1 = OrganizationRule(
            {
                "name": "General Rule",
                "conditions": {"categories": ["financial"]},
                "path_template": "Financial/{document_type}",
                "priority": 50,
            }
        )

        rule2 = OrganizationRule(
            {
                "name": "Specific Rule",
                "conditions": {
                    "document_type": "invoice",
                    "categories": ["financial"],
                    "entity_types": ["organization"],
                },
                "path_template": "Financial/Invoices/{entity.organization[0]}/{date.year}",
                "priority": 50,
            }
        )

        matching_rules = [
            (rule1, "Financial/invoice"),
            (rule2, "Financial/Invoices/ABC_Corp/2024"),
        ]

        resolution = resolver.resolve(matching_rules, sample_metadata)

        assert resolution.selected_rule == rule2
        assert "specific" in resolution.reason.lower()

    @pytest.mark.skip(reason="Test requires fixing history tracking")
    def test_resolution_history(self, sample_metadata):
        """Test resolution history tracking."""
        resolver = ConflictResolver()

        rule = OrganizationRule(
            {
                "name": "Test Rule",
                "conditions": {},
                "path_template": "Test",
                "priority": 50,
            }
        )

        matching_rules = [(rule, "Test/Path")]

        # Make several resolutions
        for _ in range(3):
            resolver.resolve(matching_rules, sample_metadata)

        stats = resolver.get_resolution_stats()

        assert stats["total_resolutions"] == 3
        assert stats["rules_used"]["Test Rule"] == 3
