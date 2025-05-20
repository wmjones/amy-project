"""
Rule engine for evaluating and applying organization rules.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class RuleEngine:
    """Engine for evaluating metadata against organization rules."""

    def __init__(self):
        """Initialize rule engine."""
        self.operators = {
            "$eq": self._op_eq,
            "$ne": self._op_ne,
            "$gt": self._op_gt,
            "$gte": self._op_gte,
            "$lt": self._op_lt,
            "$lte": self._op_lte,
            "$in": self._op_in,
            "$nin": self._op_nin,
            "$exists": self._op_exists,
            "$regex": self._op_regex,
            "$contains": self._op_contains,
            "$starts_with": self._op_starts_with,
            "$ends_with": self._op_ends_with,
        }

    def evaluate_rule(self, rule: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """
        Evaluate if metadata matches rule conditions.

        Args:
            rule: Organization rule with conditions
            metadata: File metadata to evaluate

        Returns:
            True if metadata matches rule conditions
        """
        if not rule.get("enabled", True):
            return False

        conditions = rule.get("conditions", {})
        return self._evaluate_conditions(conditions, metadata)

    def _evaluate_conditions(
        self, conditions: Dict[str, Any], metadata: Dict[str, Any]
    ) -> bool:
        """Evaluate conditions against metadata."""
        for field, condition in conditions.items():
            # Get the field value from metadata using dot notation
            value = self._get_nested_value(metadata, field)

            # Handle wildcard matching
            if condition == "*":
                if value is None:
                    return False
                continue

            # Handle operator-based conditions
            if isinstance(condition, dict):
                if not self._evaluate_operators(value, condition):
                    return False
            else:
                # Direct value comparison
                if value != condition:
                    return False

        return True

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        parts = path.split(".")
        current = data

        for part in parts:
            # Handle array indexing
            if "[" in part and "]" in part:
                field, index = part.split("[")
                index = int(index.rstrip("]"))

                if field and field in current:
                    current = current[field]

                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None

        return current

    def _evaluate_operators(self, value: Any, conditions: Dict[str, Any]) -> bool:
        """Evaluate operator-based conditions."""
        for operator, expected in conditions.items():
            if operator in self.operators:
                if not self.operators[operator](value, expected):
                    return False
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        return True

    # Operator implementations
    def _op_eq(self, value: Any, expected: Any) -> bool:
        """Equal operator."""
        return value == expected

    def _op_ne(self, value: Any, expected: Any) -> bool:
        """Not equal operator."""
        return value != expected

    def _op_gt(self, value: Any, expected: Any) -> bool:
        """Greater than operator."""
        try:
            return value > expected
        except TypeError:
            return False

    def _op_gte(self, value: Any, expected: Any) -> bool:
        """Greater than or equal operator."""
        try:
            return value >= expected
        except TypeError:
            return False

    def _op_lt(self, value: Any, expected: Any) -> bool:
        """Less than operator."""
        try:
            return value < expected
        except TypeError:
            return False

    def _op_lte(self, value: Any, expected: Any) -> bool:
        """Less than or equal operator."""
        try:
            return value <= expected
        except TypeError:
            return False

    def _op_in(self, value: Any, expected: List[Any]) -> bool:
        """In operator."""
        return value in expected

    def _op_nin(self, value: Any, expected: List[Any]) -> bool:
        """Not in operator."""
        return value not in expected

    def _op_exists(self, value: Any, expected: bool) -> bool:
        """Exists operator."""
        exists = value is not None
        return exists == expected

    def _op_regex(self, value: Any, expected: str) -> bool:
        """Regex operator."""
        if not isinstance(value, str):
            return False
        try:
            return bool(re.search(expected, value))
        except re.error:
            logger.error(f"Invalid regex pattern: {expected}")
            return False

    def _op_contains(self, value: Any, expected: Any) -> bool:
        """Contains operator."""
        if isinstance(value, str) and isinstance(expected, str):
            return expected in value
        elif isinstance(value, list):
            return expected in value
        return False

    def _op_starts_with(self, value: Any, expected: str) -> bool:
        """Starts with operator."""
        if not isinstance(value, str):
            return False
        return value.startswith(expected)

    def _op_ends_with(self, value: Any, expected: str) -> bool:
        """Ends with operator."""
        if not isinstance(value, str):
            return False
        return value.endswith(expected)

    def apply_path_template(
        self, template: str, metadata: Dict[str, Any], filename: str
    ) -> Path:
        """
        Apply path template to generate organized file path.

        Args:
            template: Path template with placeholders
            metadata: File metadata
            filename: Original filename

        Returns:
            Generated file path
        """
        path = template

        # Add filename to metadata for template processing
        metadata_with_filename = metadata.copy()
        metadata_with_filename["filename"] = filename

        # Find all placeholders in the template
        placeholders = re.findall(r"\{([^}]+)\}", template)

        for placeholder in placeholders:
            if "|" in placeholder:
                # Format specified
                field, format_spec = placeholder.split("|", 1)
                value = self._get_nested_value(metadata_with_filename, field)

                if value is not None:
                    formatted_value = self._format_value(value, format_spec)
                else:
                    formatted_value = "unknown"
            else:
                # No format specified
                value = self._get_nested_value(metadata_with_filename, placeholder)
                formatted_value = str(value) if value is not None else "unknown"

            # Replace placeholder with formatted value
            path = path.replace(f"{{{placeholder}}}", formatted_value)

        # Clean up the path
        path = self._clean_path(path)

        return Path(path)

    def _format_value(self, value: Any, format_spec: str) -> str:
        """Format a value according to format specification."""
        if isinstance(value, datetime):
            try:
                return value.strftime(format_spec)
            except:
                return str(value)
        elif isinstance(value, str):
            # Try to parse as date if format spec looks like date format
            if "%" in format_spec:
                try:
                    date_value = datetime.fromisoformat(value)
                    return date_value.strftime(format_spec)
                except:
                    pass

        # Default formatting
        try:
            return f"{value:{format_spec}}"
        except:
            return str(value)

    def _clean_path(self, path: str) -> str:
        """Clean and sanitize a file path."""
        # Replace invalid characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            path = path.replace(char, "_")

        # Remove duplicate slashes
        path = re.sub(r"/+", "/", path)

        # Remove leading/trailing slashes
        path = path.strip("/")

        return path

    def find_matching_rule(
        self, rules: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the first matching rule for given metadata.

        Args:
            rules: List of organization rules (sorted by priority)
            metadata: File metadata

        Returns:
            First matching rule or None
        """
        for rule in rules:
            if self.evaluate_rule(rule, metadata):
                logger.info(f"Matched rule: {rule['name']}")
                return rule

        return None

    def validate_path_template(self, template: str) -> List[str]:
        """
        Validate a path template and return any errors.

        Args:
            template: Path template to validate

        Returns:
            List of error messages
        """
        errors = []

        # Check for invalid characters (but not within placeholders)
        # First, temporarily remove placeholders
        temp_template = template
        placeholders = re.findall(r"\{[^}]+\}", template)
        for i, placeholder in enumerate(placeholders):
            temp_template = temp_template.replace(placeholder, f"__PLACEHOLDER_{i}__")

        # Now check for invalid characters in the template (outside placeholders)
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            if char in temp_template:
                errors.append(f"Template contains invalid character: {char}")

        # Check for balanced braces
        open_braces = template.count("{")
        close_braces = template.count("}")
        if open_braces != close_braces:
            errors.append("Unbalanced braces in template")

        # Check placeholder format
        placeholders = re.findall(r"\{([^}]*)\}", template)
        for placeholder in placeholders:
            if not placeholder:
                errors.append("Empty placeholder found")
            elif placeholder.count("|") > 1:
                errors.append(f"Invalid placeholder format: {placeholder}")

        return errors
