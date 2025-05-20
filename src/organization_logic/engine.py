"""
Organization logic engine for determining file structure.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import copy

from src.metadata_extraction.extractor import DocumentMetadata

logger = logging.getLogger(__name__)


class OrganizationRule:
    """Represents a single organization rule."""

    def __init__(self, rule_dict: Dict[str, Any]):
        """Initialize rule from dictionary.

        Args:
            rule_dict: Dictionary containing rule definition
        """
        self.name = rule_dict.get("name", "Unnamed Rule")
        self.conditions = rule_dict.get("conditions", {})
        self.path_template = rule_dict.get("path_template", "")
        self.priority = rule_dict.get("priority", 50)
        self.enabled = rule_dict.get("enabled", True)
        self.description = rule_dict.get("description", "")

    def matches(self, metadata: DocumentMetadata) -> bool:
        """Check if rule matches the given metadata.

        Args:
            metadata: Document metadata to check

        Returns:
            True if rule matches
        """
        if not self.enabled:
            return False

        # Check document type
        if "document_type" in self.conditions:
            required_type = self.conditions["document_type"]
            if isinstance(required_type, list):
                if metadata.document_type not in required_type:
                    return False
            else:
                if metadata.document_type != required_type:
                    return False

        # Check categories
        if "categories" in self.conditions:
            required_categories = self.conditions["categories"]
            if not any(cat in metadata.categories for cat in required_categories):
                return False

        # Check tags
        if "tags" in self.conditions:
            required_tags = self.conditions["tags"]
            if not any(tag in metadata.tags for tag in required_tags):
                return False

        # Check confidence threshold
        if "min_confidence" in self.conditions:
            if metadata.confidence_score < self.conditions["min_confidence"]:
                return False

        # Check date range
        if "date_after" in self.conditions or "date_before" in self.conditions:
            if not metadata.dates.document_date:
                return False

            doc_date = datetime.fromisoformat(metadata.dates.document_date)

            if "date_after" in self.conditions:
                min_date = datetime.fromisoformat(self.conditions["date_after"])
                if doc_date < min_date:
                    return False

            if "date_before" in self.conditions:
                max_date = datetime.fromisoformat(self.conditions["date_before"])
                if doc_date > max_date:
                    return False

        # Check entity types
        if "entity_types" in self.conditions:
            required_types = self.conditions["entity_types"]
            entity_types = {e.type for e in metadata.entities}
            if not any(etype in entity_types for etype in required_types):
                return False

        # Check custom conditions using expressions
        if "custom" in self.conditions:
            try:
                # Create safe evaluation context
                context = {"metadata": metadata, "len": len, "any": any, "all": all}
                result = eval(self.conditions["custom"], {"__builtins__": {}}, context)
                if not result:
                    return False
            except Exception as e:
                logger.warning(f"Error evaluating custom condition: {e}")
                return False

        return True


class OrganizationEngine:
    """Engine for determining file organization structure."""

    DEFAULT_RULES = [
        {
            "name": "Financial Documents by Year",
            "conditions": {"categories": ["financial"], "min_confidence": 0.7},
            "path_template": "Financial/{document_type}/{date.year}",
            "priority": 80,
        },
        {
            "name": "Legal Documents",
            "conditions": {"categories": ["legal"], "min_confidence": 0.7},
            "path_template": "Legal/{document_type}/{date.year}",
            "priority": 80,
        },
        {
            "name": "Medical Records by Person",
            "conditions": {
                "categories": ["medical", "health"],
                "entity_types": ["person"],
            },
            "path_template": "Medical/{entity.person[0]}/{date.year}",
            "priority": 85,
        },
        {
            "name": "Invoices by Company",
            "conditions": {
                "document_type": "invoice",
                "entity_types": ["organization"],
            },
            "path_template": "Invoices/{entity.organization[0]}/{date.year}/{date.month}",
            "priority": 90,
        },
        {
            "name": "Personal Documents",
            "conditions": {"categories": ["personal"]},
            "path_template": "Personal/{document_type}/{date.year}",
            "priority": 70,
        },
        {
            "name": "Work Documents",
            "conditions": {"categories": ["work", "business"]},
            "path_template": "Work/{document_type}/{date.year}",
            "priority": 75,
        },
        {
            "name": "General by Type",
            "conditions": {},
            "path_template": "Documents/{document_type}/{date.year}",
            "priority": 10,
        },
    ]

    def __init__(
        self,
        rules_path: Optional[str] = None,
        use_default_rules: bool = True,
        default_folder: str = "Unsorted",
    ):
        """Initialize organization engine.

        Args:
            rules_path: Optional path to custom rules file
            use_default_rules: Whether to include default rules
            default_folder: Default folder for unmatched files
        """
        self.default_folder = default_folder
        self.rules = []

        # Load default rules if requested
        if use_default_rules:
            self.rules.extend([OrganizationRule(r) for r in self.DEFAULT_RULES])

        # Load custom rules if provided
        if rules_path:
            custom_rules = self._load_rules_from_file(rules_path)
            self.rules.extend(custom_rules)

        # Sort rules by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info(f"Organization engine initialized with {len(self.rules)} rules")

    def determine_target_location(self, metadata: DocumentMetadata) -> Tuple[str, str]:
        """Determine target location for a file based on metadata.

        Args:
            metadata: Document metadata

        Returns:
            Tuple of (target_path, rule_name)
        """
        # Check Claude's suggested folder first if confidence is high
        if metadata.suggested_folder and metadata.confidence_score >= 0.85:
            logger.info(f"Using Claude's suggested folder: {metadata.suggested_folder}")
            return (metadata.suggested_folder, "Claude AI Suggestion")

        # Apply organization rules
        matching_rules = []

        for rule in self.rules:
            if rule.matches(metadata):
                path = self._generate_path(rule, metadata)
                if path:
                    matching_rules.append((rule, path))
                    logger.debug(f"Rule '{rule.name}' matched, generated path: {path}")

        # Handle results
        if matching_rules:
            # Use highest priority rule
            best_rule, best_path = matching_rules[0]
            logger.info(f"Using rule '{best_rule.name}' with path: {best_path}")
            return (best_path, best_rule.name)
        else:
            # No rules matched, use default
            default_path = self._get_default_path(metadata)
            logger.info(f"No rules matched, using default path: {default_path}")
            return (default_path, "Default Organization")

    def _generate_path(
        self, rule: OrganizationRule, metadata: DocumentMetadata
    ) -> Optional[str]:
        """Generate folder path from rule template and metadata.

        Args:
            rule: Organization rule
            metadata: Document metadata

        Returns:
            Generated path or None if generation fails
        """
        try:
            path_template = rule.path_template

            # Replace document_type placeholder
            path_template = path_template.replace(
                "{document_type}", self._sanitize_path_component(metadata.document_type)
            )

            # Replace date placeholders
            if metadata.dates.document_date:
                date = datetime.fromisoformat(metadata.dates.document_date)
                path_template = path_template.replace("{date.year}", str(date.year))
                path_template = path_template.replace(
                    "{date.month}", f"{date.month:02d}"
                )
                path_template = path_template.replace("{date.day}", f"{date.day:02d}")
                path_template = path_template.replace(
                    "{date.quarter}", f"Q{(date.month-1)//3 + 1}"
                )

            # Replace entity placeholders
            entities_by_type = {}
            for entity in metadata.entities:
                if entity.type not in entities_by_type:
                    entities_by_type[entity.type] = []
                entities_by_type[entity.type].append(entity.name)

            for entity_type, names in entities_by_type.items():
                if names:
                    placeholder = f"{{entity.{entity_type}[0]}}"
                    if placeholder in path_template:
                        path_template = path_template.replace(
                            placeholder, self._sanitize_path_component(names[0])
                        )

            # Replace category placeholder
            if metadata.categories:
                path_template = path_template.replace(
                    "{category}", self._sanitize_path_component(metadata.categories[0])
                )

            # Replace tag placeholder
            if metadata.tags:
                path_template = path_template.replace(
                    "{tag}", self._sanitize_path_component(metadata.tags[0])
                )

            # Handle custom template functions
            path_template = self._process_template_functions(path_template, metadata)

            # Remove any remaining placeholders
            path_template = re.sub(r"\{[^}]+\}", "", path_template)

            # Clean up path
            path_parts = path_template.split("/")
            path_parts = [p for p in path_parts if p]  # Remove empty parts

            return "/".join(path_parts)

        except Exception as e:
            logger.error(f"Error generating path from template: {e}")
            return None

    def _process_template_functions(
        self, template: str, metadata: DocumentMetadata
    ) -> str:
        """Process template functions like date formatting.

        Args:
            template: Path template
            metadata: Document metadata

        Returns:
            Processed template
        """
        # Process date formatting functions
        date_format_pattern = r"\{dates\.document_date\|([^}]+)\}"
        matches = re.findall(date_format_pattern, template)

        for format_string in matches:
            if metadata.dates.document_date:
                try:
                    date = datetime.fromisoformat(metadata.dates.document_date)
                    formatted_date = date.strftime(format_string)
                    template = template.replace(
                        f"{{dates.document_date|{format_string}}}", formatted_date
                    )
                except Exception as e:
                    logger.warning(f"Error formatting date: {e}")

        return template

    def _sanitize_path_component(self, component: str) -> str:
        """Sanitize a path component to be filesystem-safe.

        Args:
            component: Path component to sanitize

        Returns:
            Sanitized component
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"|?*\0'
        for char in invalid_chars:
            component = component.replace(char, "")

        # Replace spaces with underscores (optional)
        component = component.replace(" ", "_")

        # Remove leading/trailing dots and spaces
        component = component.strip(". ")

        # Limit length
        max_length = 50
        if len(component) > max_length:
            component = component[:max_length]

        return component or "unnamed"

    def _get_default_path(self, metadata: DocumentMetadata) -> str:
        """Get default path for files that don't match any rules.

        Args:
            metadata: Document metadata

        Returns:
            Default path
        """
        # Try to use basic organization
        if metadata.document_type and metadata.document_type != "unknown":
            base_path = f"{self.default_folder}/{metadata.document_type}"
        elif metadata.categories:
            base_path = f"{self.default_folder}/{metadata.categories[0]}"
        else:
            base_path = self.default_folder

        # Add year if available
        if metadata.dates.document_date:
            try:
                date = datetime.fromisoformat(metadata.dates.document_date)
                base_path = f"{base_path}/{date.year}"
            except Exception:
                pass

        return base_path

    def _load_rules_from_file(self, rules_path: str) -> List[OrganizationRule]:
        """Load organization rules from JSON file.

        Args:
            rules_path: Path to rules file

        Returns:
            List of OrganizationRule objects
        """
        try:
            with open(rules_path, "r") as f:
                rules_data = json.load(f)

            rules = []
            for rule_dict in rules_data:
                try:
                    rule = OrganizationRule(rule_dict)
                    rules.append(rule)
                except Exception as e:
                    logger.error(
                        f"Error loading rule '{rule_dict.get('name', 'Unknown')}': {e}"
                    )

            return rules

        except Exception as e:
            logger.error(f"Error loading rules from {rules_path}: {e}")
            return []

    def save_rules(self, output_path: str):
        """Save current rules to file.

        Args:
            output_path: Path to save rules
        """
        rules_data = []

        for rule in self.rules:
            rule_dict = {
                "name": rule.name,
                "conditions": rule.conditions,
                "path_template": rule.path_template,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "description": rule.description,
            }
            rules_data.append(rule_dict)

        with open(output_path, "w") as f:
            json.dump(rules_data, f, indent=2)

    def add_rule(self, rule_dict: Dict[str, Any]):
        """Add a new organization rule.

        Args:
            rule_dict: Rule definition dictionary
        """
        rule = OrganizationRule(rule_dict)
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was removed
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < initial_count

    def get_rule_by_name(self, rule_name: str) -> Optional[OrganizationRule]:
        """Get a rule by name.

        Args:
            rule_name: Name of rule

        Returns:
            Rule object or None
        """
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def test_rules(self, metadata: DocumentMetadata) -> List[Tuple[str, str]]:
        """Test all rules against metadata to see which would match.

        Args:
            metadata: Document metadata

        Returns:
            List of (rule_name, generated_path) tuples
        """
        results = []

        for rule in self.rules:
            if rule.matches(metadata):
                path = self._generate_path(rule, metadata)
                if path:
                    results.append((rule.name, path))

        return results
