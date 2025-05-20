"""
Conflict resolution for organization rules.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from src.metadata_extraction.extractor import DocumentMetadata
from .engine import OrganizationRule

logger = logging.getLogger(__name__)


@dataclass
class ConflictResolution:
    """Result of conflict resolution."""

    selected_rule: OrganizationRule
    selected_path: str
    reason: str
    alternatives: List[Tuple[OrganizationRule, str]]


class ConflictResolver:
    """Resolve conflicts when multiple organization rules match."""

    def __init__(self, strategy: str = "highest_priority"):
        """Initialize conflict resolver.

        Args:
            strategy: Resolution strategy to use
                - "highest_priority": Select rule with highest priority
                - "most_specific": Select most specific rule
                - "confidence_weighted": Weight by confidence score
                - "interactive": Ask user to choose
        """
        self.strategy = strategy
        self.resolution_history = []

    def resolve(
        self,
        matching_rules: List[Tuple[OrganizationRule, str]],
        metadata: DocumentMetadata,
    ) -> ConflictResolution:
        """Resolve conflict between multiple matching rules.

        Args:
            matching_rules: List of (rule, generated_path) tuples
            metadata: Document metadata

        Returns:
            ConflictResolution object
        """
        if not matching_rules:
            raise ValueError("No matching rules to resolve")

        if len(matching_rules) == 1:
            rule, path = matching_rules[0]
            return ConflictResolution(
                selected_rule=rule,
                selected_path=path,
                reason="Only one rule matched",
                alternatives=[],
            )

        # Apply resolution strategy
        if self.strategy == "highest_priority":
            resolution = self._resolve_by_priority(matching_rules, metadata)
        elif self.strategy == "most_specific":
            resolution = self._resolve_by_specificity(matching_rules, metadata)
        elif self.strategy == "confidence_weighted":
            resolution = self._resolve_by_confidence(matching_rules, metadata)
        else:
            # Default to highest priority
            resolution = self._resolve_by_priority(matching_rules, metadata)

        # Log resolution
        self._log_resolution(resolution, metadata)

        return resolution

    def _resolve_by_priority(
        self,
        matching_rules: List[Tuple[OrganizationRule, str]],
        metadata: DocumentMetadata,
    ) -> ConflictResolution:
        """Resolve by selecting highest priority rule."""
        # Rules are already sorted by priority (highest first)
        selected_rule, selected_path = matching_rules[0]

        return ConflictResolution(
            selected_rule=selected_rule,
            selected_path=selected_path,
            reason=f"Highest priority ({selected_rule.priority})",
            alternatives=matching_rules[1:],
        )

    def _resolve_by_specificity(
        self,
        matching_rules: List[Tuple[OrganizationRule, str]],
        metadata: DocumentMetadata,
    ) -> ConflictResolution:
        """Resolve by selecting most specific rule."""
        # Score rules by specificity
        scored_rules = []

        for rule, path in matching_rules:
            score = self._calculate_specificity_score(rule, metadata)
            scored_rules.append((score, rule, path))

        # Sort by specificity score (descending)
        scored_rules.sort(key=lambda x: x[0], reverse=True)

        # Select most specific
        _, selected_rule, selected_path = scored_rules[0]

        alternatives = [(r, p) for _, r, p in scored_rules[1:]]

        return ConflictResolution(
            selected_rule=selected_rule,
            selected_path=selected_path,
            reason=f"Most specific rule (score: {scored_rules[0][0]})",
            alternatives=alternatives,
        )

    def _resolve_by_confidence(
        self,
        matching_rules: List[Tuple[OrganizationRule, str]],
        metadata: DocumentMetadata,
    ) -> ConflictResolution:
        """Resolve by weighting rule priority with metadata confidence."""
        # Score rules by priority * confidence
        scored_rules = []

        for rule, path in matching_rules:
            score = rule.priority * metadata.confidence_score
            scored_rules.append((score, rule, path))

        # Sort by weighted score (descending)
        scored_rules.sort(key=lambda x: x[0], reverse=True)

        # Select highest scoring
        _, selected_rule, selected_path = scored_rules[0]

        alternatives = [(r, p) for _, r, p in scored_rules[1:]]

        return ConflictResolution(
            selected_rule=selected_rule,
            selected_path=selected_path,
            reason=f"Confidence-weighted priority (score: {scored_rules[0][0]:.2f})",
            alternatives=alternatives,
        )

    def _calculate_specificity_score(
        self, rule: OrganizationRule, metadata: DocumentMetadata
    ) -> int:
        """Calculate specificity score for a rule.

        Args:
            rule: Organization rule
            metadata: Document metadata

        Returns:
            Specificity score (higher is more specific)
        """
        score = 0
        conditions = rule.conditions

        # More conditions = more specific
        score += len(conditions) * 10

        # Specific condition types have different weights
        if "document_type" in conditions:
            score += 20

        if "entity_types" in conditions:
            score += 15

        if "custom" in conditions:
            score += 25  # Custom conditions are very specific

        if "date_after" in conditions or "date_before" in conditions:
            score += 10

        if "min_confidence" in conditions:
            # Higher confidence requirement = more specific
            score += int(conditions["min_confidence"] * 20)

        # Path template complexity (more placeholders = more specific)
        path_placeholders = rule.path_template.count("{")
        score += path_placeholders * 5

        return score

    def _log_resolution(
        self, resolution: ConflictResolution, metadata: DocumentMetadata
    ):
        """Log conflict resolution for debugging and analysis."""
        log_entry = {
            "timestamp": metadata.processing_timestamp,
            "file": metadata.source_file,
            "selected_rule": resolution.selected_rule.name,
            "selected_path": resolution.selected_path,
            "reason": resolution.reason,
            "alternatives": [(r.name, p) for r, p in resolution.alternatives],
        }

        self.resolution_history.append(log_entry)

        logger.info(
            f"Resolved conflict for {metadata.source_file}: "
            f"selected '{resolution.selected_rule.name}' ({resolution.reason})"
        )

    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get statistics about conflict resolutions.

        Returns:
            Dictionary with resolution statistics
        """
        if not self.resolution_history:
            return {"total_resolutions": 0, "rules_used": {}, "reasons": {}}

        stats = {
            "total_resolutions": len(self.resolution_history),
            "rules_used": {},
            "reasons": {},
        }

        for entry in self.resolution_history:
            # Count rule usage
            rule_name = entry["selected_rule"]
            stats["rules_used"][rule_name] = stats["rules_used"].get(rule_name, 0) + 1

            # Count reason types
            reason = entry["reason"]
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

        return stats

    def export_resolution_history(self, output_path: str):
        """Export resolution history to file.

        Args:
            output_path: Path for output file
        """
        import json

        with open(output_path, "w") as f:
            json.dump(self.resolution_history, f, indent=2)

    def clear_history(self):
        """Clear resolution history."""
        self.resolution_history = []


class InteractiveResolver(ConflictResolver):
    """Interactive conflict resolver that asks for user input."""

    def __init__(self):
        """Initialize interactive resolver."""
        super().__init__(strategy="interactive")

    def resolve(
        self,
        matching_rules: List[Tuple[OrganizationRule, str]],
        metadata: DocumentMetadata,
    ) -> ConflictResolution:
        """Resolve conflict by asking user to choose."""
        if len(matching_rules) == 1:
            rule, path = matching_rules[0]
            return ConflictResolution(
                selected_rule=rule,
                selected_path=path,
                reason="Only one rule matched",
                alternatives=[],
            )

        # Display options to user
        print(f"\nMultiple organization rules match '{metadata.source_file}':")
        print(f"Document type: {metadata.document_type}")
        print(f"Categories: {', '.join(metadata.categories)}")
        print(f"Confidence: {metadata.confidence_score:.2f}")
        print("\nMatching rules:")

        for i, (rule, path) in enumerate(matching_rules):
            print(f"{i+1}. {rule.name} (priority: {rule.priority})")
            print(f"   Path: {path}")
            print(f"   Description: {rule.description}")
            print()

        # Get user choice
        while True:
            try:
                choice = input("Select rule (enter number): ")
                index = int(choice) - 1

                if 0 <= index < len(matching_rules):
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Create resolution
        selected_rule, selected_path = matching_rules[index]
        alternatives = matching_rules[:index] + matching_rules[index + 1 :]

        return ConflictResolution(
            selected_rule=selected_rule,
            selected_path=selected_path,
            reason="User selection",
            alternatives=alternatives,
        )
