#!/usr/bin/env python3
"""
Demo script showing organization logic engine.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.organization_logic.engine import OrganizationEngine, OrganizationRule
from src.organization_logic.rule_manager import RuleManager
from src.organization_logic.conflict_resolver import ConflictResolver
from src.metadata_extraction.extractor import DocumentMetadata, DateInfo, Entity

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_basic_organization():
    """Demonstrate basic organization with default rules."""
    print("=== Basic Organization Demo ===\n")

    # Create engine with default rules
    engine = OrganizationEngine(use_default_rules=True)

    # Sample metadata for different document types
    test_documents = [
        DocumentMetadata(
            document_type="invoice",
            categories=["financial"],
            dates=DateInfo(document_date="2024-03-15"),
            entities=[Entity(name="ABC Corporation", type="organization")],
            topics=["payment", "services"],
            tags=["Q1-2024"],
            summary="Invoice for consulting services",
            suggested_folder="",
            confidence_score=0.92,
            source_file="invoice_12345.pdf",
            processing_timestamp=datetime.now().isoformat(),
        ),
        DocumentMetadata(
            document_type="contract",
            categories=["legal"],
            dates=DateInfo(document_date="2024-01-01"),
            entities=[
                Entity(name="John Doe", type="person"),
                Entity(name="XYZ Corp", type="organization"),
            ],
            topics=["employment", "agreement"],
            tags=["2024", "employment"],
            summary="Employment contract",
            suggested_folder="",
            confidence_score=0.88,
            source_file="employment_contract.pdf",
            processing_timestamp=datetime.now().isoformat(),
        ),
        DocumentMetadata(
            document_type="medical_record",
            categories=["medical", "health"],
            dates=DateInfo(document_date="2024-02-20"),
            entities=[Entity(name="Jane Smith", type="person")],
            topics=["checkup", "health"],
            tags=["annual", "2024"],
            summary="Annual health checkup report",
            suggested_folder="",
            confidence_score=0.85,
            source_file="health_report.pdf",
            processing_timestamp=datetime.now().isoformat(),
        ),
    ]

    # Determine organization for each document
    for metadata in test_documents:
        print(f"\nDocument: {metadata.source_file}")
        print(f"Type: {metadata.document_type}")
        print(f"Categories: {', '.join(metadata.categories)}")

        # Get target location
        path, rule_name = engine.determine_target_location(metadata)

        print(f"Organization Rule: {rule_name}")
        print(f"Target Path: {path}")

        # Show which rules matched
        matching_rules = engine.test_rules(metadata)
        if len(matching_rules) > 1:
            print(f"Other matching rules:")
            for rule_name, rule_path in matching_rules:
                print(f"  - {rule_name}: {rule_path}")


def demo_custom_rules():
    """Demonstrate custom organization rules."""
    print("\n\n=== Custom Rules Demo ===\n")

    # Create custom rules
    custom_rules = [
        {
            "name": "Project Documents",
            "conditions": {"tags": ["project"], "min_confidence": 0.7},
            "path_template": "Projects/{tags[0]}/{date.year}/{document_type}",
            "priority": 90,
            "description": "Organize project-related documents",
        },
        {
            "name": "Client Invoices",
            "conditions": {
                "document_type": "invoice",
                "entity_types": ["organization"],
                "custom": "metadata.confidence_score > 0.8",
            },
            "path_template": "Clients/{entity.organization[0]}/Invoices/{date.year}",
            "priority": 95,
            "description": "Organize invoices by client",
        },
        {
            "name": "Personal Photos",
            "conditions": {
                "document_type": ["image", "photo"],
                "categories": ["personal"],
            },
            "path_template": "Photos/{date.year}/{date.month}",
            "priority": 70,
            "description": "Organize personal photos by date",
        },
    ]

    # Create engine with custom rules only
    engine = OrganizationEngine(use_default_rules=False)

    # Add custom rules
    for rule_dict in custom_rules:
        engine.add_rule(rule_dict)

    # Test with sample metadata
    test_metadata = DocumentMetadata(
        document_type="invoice",
        categories=["financial"],
        dates=DateInfo(document_date="2024-03-15"),
        entities=[Entity(name="Acme Inc", type="organization")],
        topics=["services"],
        tags=["project", "alpha"],
        summary="Invoice for Project Alpha",
        suggested_folder="",
        confidence_score=0.85,
        source_file="project_invoice.pdf",
        processing_timestamp=datetime.now().isoformat(),
    )

    print(f"Document: {test_metadata.source_file}")

    # Test all rules
    matching_rules = engine.test_rules(test_metadata)
    print(f"\nMatching rules ({len(matching_rules)}):")
    for rule_name, path in matching_rules:
        print(f"  - {rule_name}: {path}")

    # Determine final location
    path, rule_name = engine.determine_target_location(test_metadata)
    print(f"\nSelected rule: {rule_name}")
    print(f"Final path: {path}")


def demo_rule_manager():
    """Demonstrate rule management."""
    print("\n\n=== Rule Manager Demo ===\n")

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = RuleManager(temp_dir)

        # Generate example rules
        example_rules = manager.generate_example_rules()
        print(f"Generated {len(example_rules)} example rules:")

        for rule in example_rules[:3]:  # Show first 3
            print(f"\n{rule['name']}:")
            print(f"  Priority: {rule['priority']}")
            print(f"  Path: {rule['path_template']}")
            print(f"  Conditions: {json.dumps(rule['conditions'], indent=4)}")

        # Save rules
        manager.save_rule_set(example_rules, "example_rules")
        print(f"\n✅ Saved rule set: example_rules")

        # List available rule sets
        available_sets = manager.list_rule_sets()
        print(f"\nAvailable rule sets: {available_sets}")

        # Create rule from template
        print("\n\nCreating custom rule from template:")
        custom_rule = manager.create_rule_from_template(
            "financial_quarterly",
            {
                "name": "Quarterly Financial Reports",
                "conditions.categories": ["financial", "report"],
                "priority": 88,
            },
        )

        print(json.dumps(custom_rule, indent=2))


def demo_conflict_resolution():
    """Demonstrate conflict resolution."""
    print("\n\n=== Conflict Resolution Demo ===\n")

    # Create conflicting rules
    rule1 = OrganizationRule(
        {
            "name": "General Financial",
            "conditions": {"categories": ["financial"]},
            "path_template": "Financial/{document_type}/{date.year}",
            "priority": 60,
        }
    )

    rule2 = OrganizationRule(
        {
            "name": "Specific Invoices",
            "conditions": {"document_type": "invoice", "categories": ["financial"]},
            "path_template": "Invoices/{entity.organization[0]}/{date.year}",
            "priority": 80,
        }
    )

    rule3 = OrganizationRule(
        {
            "name": "High Confidence Financial",
            "conditions": {"categories": ["financial"], "min_confidence": 0.9},
            "path_template": "Verified/Financial/{document_type}",
            "priority": 85,
        }
    )

    # Test metadata that matches all rules
    metadata = DocumentMetadata(
        document_type="invoice",
        categories=["financial"],
        dates=DateInfo(document_date="2024-03-15"),
        entities=[Entity(name="TechCorp", type="organization")],
        topics=["services"],
        tags=["Q1-2024"],
        summary="Monthly invoice",
        suggested_folder="",
        confidence_score=0.95,
        source_file="techcorp_invoice.pdf",
        processing_timestamp=datetime.now().isoformat(),
    )

    # Simulate matches
    matching_rules = [
        (rule3, "Verified/Financial/invoice"),
        (rule2, "Invoices/TechCorp/2024"),
        (rule1, "Financial/invoice/2024"),
    ]

    # Test different resolution strategies
    strategies = ["highest_priority", "most_specific", "confidence_weighted"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        resolver = ConflictResolver(strategy=strategy)

        resolution = resolver.resolve(matching_rules, metadata)

        print(f"Selected: {resolution.selected_rule.name}")
        print(f"Path: {resolution.selected_path}")
        print(f"Reason: {resolution.reason}")

        if resolution.alternatives:
            print(f"Alternatives:")
            for rule, path in resolution.alternatives:
                print(f"  - {rule.name}: {path}")


def demo_claude_integration():
    """Demonstrate Claude AI suggestion integration."""
    print("\n\n=== Claude AI Integration Demo ===\n")

    engine = OrganizationEngine(use_default_rules=True)

    # High confidence Claude suggestion
    metadata1 = DocumentMetadata(
        document_type="report",
        categories=["business"],
        dates=DateInfo(document_date="2024-03-01"),
        entities=[],
        topics=["quarterly", "performance"],
        tags=["Q1-2024"],
        summary="Q1 performance report",
        suggested_folder="Reports/Quarterly/2024/Q1",
        confidence_score=0.92,  # High confidence
        source_file="q1_report.pdf",
        processing_timestamp=datetime.now().isoformat(),
    )

    print(f"Document: {metadata1.source_file}")
    print(f"Claude suggestion: {metadata1.suggested_folder}")
    print(f"Confidence: {metadata1.confidence_score}")

    path1, rule1 = engine.determine_target_location(metadata1)
    print(f"Result: {path1} (using {rule1})")

    # Lower confidence - use rules instead
    metadata2 = DocumentMetadata(
        document_type="report",
        categories=["business"],
        dates=DateInfo(document_date="2024-03-01"),
        entities=[],
        topics=["quarterly", "performance"],
        tags=["Q1-2024"],
        summary="Q1 performance report",
        suggested_folder="Reports/Quarterly/2024/Q1",
        confidence_score=0.75,  # Lower confidence
        source_file="q1_report_draft.pdf",
        processing_timestamp=datetime.now().isoformat(),
    )

    print(f"\nDocument: {metadata2.source_file}")
    print(f"Claude suggestion: {metadata2.suggested_folder}")
    print(f"Confidence: {metadata2.confidence_score}")

    path2, rule2 = engine.determine_target_location(metadata2)
    print(f"Result: {path2} (using {rule2})")


def main():
    """Run all demos."""
    print("Organization Logic Demo")
    print("=====================")

    demo_basic_organization()
    demo_custom_rules()
    demo_rule_manager()
    demo_conflict_resolution()
    demo_claude_integration()

    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
