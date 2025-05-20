"""
Rule management utilities for organization engine.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger(__name__)


class RuleManager:
    """Manage organization rules."""

    RULE_TEMPLATES = {
        "by_type_and_date": {
            "name": "Organize by Type and Date",
            "conditions": {"document_type": "SPECIFY_TYPE", "min_confidence": 0.7},
            "path_template": "{document_type}/{date.year}/{date.month}",
            "priority": 70,
            "description": "Organize documents by type and date",
        },
        "by_category": {
            "name": "Organize by Category",
            "conditions": {"categories": ["SPECIFY_CATEGORY"], "min_confidence": 0.7},
            "path_template": "{category}/{document_type}/{date.year}",
            "priority": 60,
            "description": "Organize documents by category",
        },
        "by_entity": {
            "name": "Organize by Entity",
            "conditions": {
                "entity_types": ["SPECIFY_ENTITY_TYPE"],
                "min_confidence": 0.7,
            },
            "path_template": "{entity.ENTITY_TYPE[0]}/{document_type}/{date.year}",
            "priority": 80,
            "description": "Organize documents by entity (person/organization)",
        },
        "financial_quarterly": {
            "name": "Financial Documents by Quarter",
            "conditions": {"categories": ["financial"], "min_confidence": 0.8},
            "path_template": "Financial/{date.year}/{date.quarter}/{document_type}",
            "priority": 85,
            "description": "Organize financial documents by quarter",
        },
    }

    def __init__(self, rules_directory: Optional[str] = None):
        """Initialize rule manager.

        Args:
            rules_directory: Directory to store rule files
        """
        if rules_directory:
            self.rules_dir = Path(rules_directory)
            self.rules_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.rules_dir = None

    def create_rule_from_template(
        self, template_name: str, customizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a rule from a template.

        Args:
            template_name: Name of template
            customizations: Custom values to apply

        Returns:
            Customized rule dictionary
        """
        if template_name not in self.RULE_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        # Deep copy template
        rule = json.loads(json.dumps(self.RULE_TEMPLATES[template_name]))

        # Apply customizations
        for key, value in customizations.items():
            # Handle nested keys (e.g., "conditions.document_type")
            keys = key.split(".")
            target = rule

            for k in keys[:-1]:
                target = target[k]

            target[keys[-1]] = value

        # Update path template if entity type was customized
        if "conditions.entity_types" in customizations:
            entity_type = customizations["conditions.entity_types"][0]
            rule["path_template"] = rule["path_template"].replace(
                "ENTITY_TYPE", entity_type
            )

        return rule

    def validate_rule(self, rule: Dict[str, Any]) -> List[str]:
        """Validate a rule definition.

        Args:
            rule: Rule dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if "name" not in rule:
            errors.append("Rule must have a 'name' field")

        if "path_template" not in rule:
            errors.append("Rule must have a 'path_template' field")

        # Validate conditions if present
        if "conditions" in rule:
            conditions = rule["conditions"]

            # Check for valid condition types
            valid_conditions = {
                "document_type",
                "categories",
                "tags",
                "min_confidence",
                "date_after",
                "date_before",
                "entity_types",
                "custom",
            }

            for condition in conditions:
                if condition not in valid_conditions:
                    errors.append(f"Unknown condition type: {condition}")

            # Validate specific condition formats
            if "min_confidence" in conditions:
                try:
                    confidence = float(conditions["min_confidence"])
                    if confidence < 0 or confidence > 1:
                        errors.append("min_confidence must be between 0 and 1")
                except (ValueError, TypeError):
                    errors.append("min_confidence must be a number")

        # Validate path template
        path_template = rule.get("path_template", "")
        if not path_template:
            errors.append("path_template cannot be empty")

        # Check for basic template syntax
        if "{" in path_template and "}" not in path_template:
            errors.append("Invalid path_template syntax: unmatched braces")

        return errors

    def save_rule_set(self, rules: List[Dict[str, Any]], name: str):
        """Save a set of rules to file.

        Args:
            rules: List of rule dictionaries
            name: Name for the rule set
        """
        if not self.rules_dir:
            raise ValueError("No rules directory configured")

        # Validate all rules
        for i, rule in enumerate(rules):
            errors = self.validate_rule(rule)
            if errors:
                raise ValueError(f"Rule {i} validation errors: {errors}")

        # Save as JSON
        file_path = self.rules_dir / f"{name}.json"
        with open(file_path, "w") as f:
            json.dump(rules, f, indent=2)

        logger.info(f"Saved {len(rules)} rules to {file_path}")

    def load_rule_set(self, name: str) -> List[Dict[str, Any]]:
        """Load a rule set from file.

        Args:
            name: Name of the rule set

        Returns:
            List of rule dictionaries
        """
        if not self.rules_dir:
            raise ValueError("No rules directory configured")

        file_path = self.rules_dir / f"{name}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Rule set not found: {name}")

        with open(file_path, "r") as f:
            rules = json.load(f)

        return rules

    def list_rule_sets(self) -> List[str]:
        """List available rule sets.

        Returns:
            List of rule set names
        """
        if not self.rules_dir:
            return []

        rule_files = self.rules_dir.glob("*.json")
        return [f.stem for f in rule_files]

    def merge_rule_sets(self, sets: List[str]) -> List[Dict[str, Any]]:
        """Merge multiple rule sets.

        Args:
            sets: List of rule set names to merge

        Returns:
            Merged list of rules
        """
        merged = []
        seen_names = set()

        for set_name in sets:
            rules = self.load_rule_set(set_name)

            for rule in rules:
                # Skip duplicates by name
                if rule["name"] not in seen_names:
                    merged.append(rule)
                    seen_names.add(rule["name"])

        return merged

    def export_rules_to_yaml(self, rules: List[Dict[str, Any]], output_path: str):
        """Export rules to YAML format.

        Args:
            rules: List of rules
            output_path: Path for YAML file
        """
        with open(output_path, "w") as f:
            yaml.dump(rules, f, default_flow_style=False)

    def import_rules_from_yaml(self, yaml_path: str) -> List[Dict[str, Any]]:
        """Import rules from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            List of rule dictionaries
        """
        with open(yaml_path, "r") as f:
            rules = yaml.safe_load(f)

        # Validate imported rules
        for i, rule in enumerate(rules):
            errors = self.validate_rule(rule)
            if errors:
                raise ValueError(f"Rule {i} validation errors: {errors}")

        return rules

    def generate_example_rules(self) -> List[Dict[str, Any]]:
        """Generate a set of example rules.

        Returns:
            List of example rules
        """
        examples = []

        # Financial invoices
        examples.append(
            self.create_rule_from_template(
                "by_type_and_date",
                {
                    "name": "Invoices by Company",
                    "conditions.document_type": "invoice",
                    "conditions.entity_types": ["organization"],
                    "path_template": "Financial/Invoices/{entity.organization[0]}/{date.year}",
                    "priority": 90,
                },
            )
        )

        # Legal contracts
        examples.append(
            self.create_rule_from_template(
                "by_category",
                {
                    "name": "Legal Contracts",
                    "conditions.categories": ["legal"],
                    "conditions.document_type": "contract",
                    "path_template": "Legal/Contracts/{entity.organization[0]}/{date.year}",
                    "priority": 85,
                },
            )
        )

        # Medical records
        examples.append(
            self.create_rule_from_template(
                "by_entity",
                {
                    "name": "Medical Records by Patient",
                    "conditions.categories": ["medical"],
                    "conditions.entity_types": ["person"],
                    "path_template": "Medical/{entity.person[0]}/{date.year}/{document_type}",
                    "priority": 85,
                },
            )
        )

        # Work documents
        examples.append(
            {
                "name": "Work Projects",
                "conditions": {
                    "categories": ["work", "project"],
                    "tags": ["project"],
                    "min_confidence": 0.7,
                },
                "path_template": "Work/Projects/{tag}/{date.year}/{date.month}",
                "priority": 75,
                "description": "Organize work project documents",
            }
        )

        # Personal documents
        examples.append(
            {
                "name": "Personal Documents",
                "conditions": {
                    "categories": ["personal"],
                    "custom": "len(metadata.entities) == 0",  # No entities (private docs)
                },
                "path_template": "Personal/{document_type}/{date.year}",
                "priority": 70,
                "description": "Personal documents without business entities",
            }
        )

        return examples
