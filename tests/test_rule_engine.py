"""
Unit tests for the rule engine.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.utils.rule_engine import RuleEngine


class TestRuleEngine:
    """Test the RuleEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RuleEngine()

        # Sample metadata
        self.metadata = {
            "document_type": "invoice",
            "dates": {"document_date": "2023-10-15", "processing_date": datetime.now()},
            "entities": {
                "people": ["John Doe", "Jane Smith"],
                "organizations": ["Acme Corp", "Tech Solutions"],
            },
            "amounts": {"total": 1500.50, "tax": 150.05},
            "tags": ["urgent", "paid"],
        }

        # Sample rules
        self.rules = [
            {
                "name": "Invoice by client",
                "priority": 1,
                "conditions": {
                    "document_type": "invoice",
                    "entities.organizations": "*",
                },
                "path_template": "Invoices/{entities.organizations[0]}/{dates.document_date|%Y}/{filename}",
                "enabled": True,
            },
            {
                "name": "High value documents",
                "priority": 2,
                "conditions": {"amounts.total": {"$gte": 1000}},
                "path_template": "High_Value/{document_type}/{filename}",
                "enabled": True,
            },
            {
                "name": "Urgent documents",
                "priority": 3,
                "conditions": {"tags": {"$contains": "urgent"}},
                "path_template": "Urgent/{document_type}/{filename}",
                "enabled": True,
            },
        ]

    def test_evaluate_simple_conditions(self):
        """Test evaluation of simple conditions."""
        # Exact match
        rule = {"conditions": {"document_type": "invoice"}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Mismatch
        rule["conditions"]["document_type"] = "receipt"
        assert self.engine.evaluate_rule(rule, self.metadata) is False

        # Wildcard match
        rule = {"conditions": {"document_type": "*"}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

    def test_evaluate_nested_conditions(self):
        """Test evaluation of nested conditions."""
        # Nested field match
        rule = {"conditions": {"dates.document_date": "2023-10-15"}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Array access
        rule = {"conditions": {"entities.people[0]": "John Doe"}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Array access out of bounds
        rule = {"conditions": {"entities.people[5]": "Someone"}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is False

    def test_evaluate_operators(self):
        """Test evaluation of operator conditions."""
        # Greater than or equal
        rule = {"conditions": {"amounts.total": {"$gte": 1000}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        rule["conditions"]["amounts.total"]["$gte"] = 2000
        assert self.engine.evaluate_rule(rule, self.metadata) is False

        # In operator
        rule = {
            "conditions": {"document_type": {"$in": ["invoice", "receipt"]}},
            "enabled": True,
        }
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Contains operator
        rule = {"conditions": {"tags": {"$contains": "urgent"}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Exists operator
        rule = {"conditions": {"amounts.total": {"$exists": True}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        rule = {
            "conditions": {"non_existent_field": {"$exists": False}},
            "enabled": True,
        }
        assert self.engine.evaluate_rule(rule, self.metadata) is True

    def test_string_operators(self):
        """Test string-specific operators."""
        metadata = {
            "filename": "invoice_2023_001.pdf",
            "client_name": "Acme Corporation",
        }

        # Starts with
        rule = {
            "conditions": {"filename": {"$starts_with": "invoice"}},
            "enabled": True,
        }
        assert self.engine.evaluate_rule(rule, metadata) is True

        # Ends with
        rule = {"conditions": {"filename": {"$ends_with": ".pdf"}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, metadata) is True

        # Regex
        rule = {
            "conditions": {"filename": {"$regex": r"invoice_\d{4}_\d{3}"}},
            "enabled": True,
        }
        assert self.engine.evaluate_rule(rule, metadata) is True

    def test_complex_conditions(self):
        """Test complex conditions with multiple criteria."""
        rule = {
            "conditions": {
                "document_type": "invoice",
                "amounts.total": {"$gte": 1000},
                "tags": {"$contains": "urgent"},
                "entities.organizations": {"$exists": True},
            },
            "enabled": True,
        }
        assert self.engine.evaluate_rule(rule, self.metadata) is True

        # Change one condition to fail
        rule["conditions"]["document_type"] = "receipt"
        assert self.engine.evaluate_rule(rule, self.metadata) is False

    def test_apply_path_template(self):
        """Test path template application."""
        template = (
            "Invoices/{entities.organizations[0]}/{dates.document_date|%Y}/{filename}"
        )
        path = self.engine.apply_path_template(
            template, self.metadata, "invoice_001.pdf"
        )

        assert path == Path("Invoices/Acme Corp/2023/invoice_001.pdf")

        # Test with datetime formatting
        template = "{document_type}/{dates.processing_date|%Y-%m-%d}/{filename}"
        path = self.engine.apply_path_template(template, self.metadata, "doc.pdf")

        today = datetime.now().strftime("%Y-%m-%d")
        assert path == Path(f"invoice/{today}/doc.pdf")

        # Test with missing field
        template = "{missing_field}/{filename}"
        path = self.engine.apply_path_template(template, self.metadata, "doc.pdf")

        assert path == Path("unknown/doc.pdf")

    def test_find_matching_rule(self):
        """Test finding the first matching rule."""
        # Should match first rule (invoice by client)
        rule = self.engine.find_matching_rule(self.rules, self.metadata)
        assert rule is not None
        assert rule["name"] == "Invoice by client"

        # Modify metadata to match only second rule
        metadata = {"document_type": "receipt", "amounts": {"total": 2000}}
        rule = self.engine.find_matching_rule(self.rules, metadata)
        assert rule is not None
        assert rule["name"] == "High value documents"

        # No matching rule
        metadata = {"document_type": "photo"}
        rule = self.engine.find_matching_rule(self.rules, metadata)
        assert rule is None

    def test_disabled_rules(self):
        """Test that disabled rules are not evaluated."""
        rule = {"conditions": {"document_type": "invoice"}, "enabled": False}
        assert self.engine.evaluate_rule(rule, self.metadata) is False

    def test_validate_path_template(self):
        """Test path template validation."""
        # Valid template
        template = "{document_type}/{dates.document_date|%Y}/{filename}"
        errors = self.engine.validate_path_template(template)
        assert len(errors) == 0

        # Invalid characters
        template = "Documents<>/{filename}"
        errors = self.engine.validate_path_template(template)
        assert any("invalid character" in error for error in errors)

        # Unbalanced braces
        template = "{document_type/{filename}"
        errors = self.engine.validate_path_template(template)
        assert any("Unbalanced braces" in error for error in errors)

        # Empty placeholder
        template = "Documents/{}/{filename}"
        errors = self.engine.validate_path_template(template)
        assert any("Empty placeholder" in error for error in errors)

        # Invalid placeholder format
        template = "{field|format|extra}/{filename}"
        errors = self.engine.validate_path_template(template)
        assert any("Invalid placeholder format" in error for error in errors)

    def test_clean_path(self):
        """Test path cleaning functionality."""
        # Test through apply_path_template
        template = "Documents//Type//{filename}"
        metadata = {"document_type": "invoice"}
        path = self.engine.apply_path_template(template, metadata, "file.pdf")

        # Should remove duplicate slashes
        assert str(path) == "Documents/Type/file.pdf"

        # Test with invalid characters (will be replaced in template)
        template = "Doc<>Type/{filename}"
        path = self.engine.apply_path_template(template, metadata, "file.pdf")
        assert "<" not in str(path) and ">" not in str(path)

    def test_operator_type_safety(self):
        """Test operators handle type mismatches safely."""
        # Compare string with number
        rule = {"conditions": {"document_type": {"$gt": 100}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is False

        # Regex on non-string
        rule = {"conditions": {"amounts.total": {"$regex": r"\d+"}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is False

        # Contains on non-container
        rule = {"conditions": {"amounts.total": {"$contains": 100}}, "enabled": True}
        assert self.engine.evaluate_rule(rule, self.metadata) is False


if __name__ == "__main__":
    pytest.main([__file__])
