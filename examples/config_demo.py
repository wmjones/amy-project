"""
Demo script showing configuration management features.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_manager import ConfigManager
from src.utils.rule_engine import RuleEngine


def demonstrate_config_loading():
    """Demonstrate configuration loading from various sources."""
    print("=== Configuration Loading Demo ===\n")
    
    # 1. Default configuration
    print("1. Loading default configuration:")
    config = ConfigManager()
    print(f"Default model: {config.get('api.claude_model')}")
    print(f"Default batch size: {config.get('processing.batch_size')}")
    print(f"Default organization mode: {config.get('organization.mode')}\n")
    
    # 2. Loading from JSON file
    print("2. Loading from config file:")
    config_file = Path(__file__).parent.parent / "config" / "config_example.json"
    if config_file.exists():
        config = ConfigManager(config_file=config_file)
        print(f"Loaded model: {config.get('api.claude_model')}")
        print(f"Loaded batch size: {config.get('processing.batch_size')}")
        print(f"Number of rules: {len(config.get_rules())}\n")
    else:
        print("Config file not found\n")
    
    # 3. Environment variable override
    print("3. Environment variable override:")
    os.environ['FILE_ORGANIZER_PROCESSING__BATCH_SIZE'] = '25'
    os.environ['FILE_ORGANIZER_LOGGING__LEVEL'] = 'DEBUG'
    config = ConfigManager()
    print(f"Batch size from env: {config.get('processing.batch_size')}")
    print(f"Log level from env: {config.get('logging.level')}\n")
    
    # 4. Command line override
    print("4. Command line argument override:")
    args = argparse.Namespace(
        source_dir='/data/source',
        output_dir='/data/output',
        batch_size=30,
        mode='move',
        max_workers=8,
        log_level='WARNING',
        config_file=None
    )
    config = ConfigManager(cli_args=args)
    print(f"Batch size from CLI: {config.get('processing.batch_size')}")
    print(f"Organization mode from CLI: {config.get('organization.mode')}")
    print(f"Log level from CLI: {config.get('logging.level')}\n")


def demonstrate_rule_management():
    """Demonstrate organization rule management."""
    print("=== Rule Management Demo ===\n")
    
    config = ConfigManager()
    
    # 1. Get default rules
    print("1. Default organization rules:")
    rules = config.get_rules()
    for i, rule in enumerate(rules[:3]):  # Show first 3 rules
        print(f"  Rule {i+1}: {rule['name']}")
        print(f"    Priority: {rule['priority']}")
        print(f"    Path template: {rule['path_template']}\n")
    
    # 2. Add custom rule
    print("2. Adding custom rule:")
    custom_rule = {
        'name': 'Tax documents by year',
        'priority': 15,
        'conditions': {
            'document_type': 'tax_document',
            'dates.tax_year': {'$exists': True}
        },
        'path_template': 'Tax/{dates.tax_year}/{document_type}/{filename}',
        'enabled': True,
        'description': 'Organize tax documents by tax year'
    }
    
    config.add_rule(custom_rule)
    print(f"Added rule: {custom_rule['name']}")
    print(f"Total rules: {len(config.get_rules())}\n")
    
    # 3. Validate rule
    print("3. Rule validation:")
    
    # Valid rule
    errors = config.validate_rule(custom_rule)
    print(f"Custom rule validation: {'OK' if not errors else 'FAILED'}")
    
    # Invalid rule
    invalid_rule = {
        'name': 'Invalid Rule',
        'conditions': {'document_type': 'test'},
        'path_template': '{missing_field}/{filename}'
    }
    errors = config.validate_rule(invalid_rule)
    print(f"Invalid rule validation: {errors[0] if errors else 'OK'}\n")
    
    # 4. Rule template
    print("4. Rule template:")
    template = config.get_rule_template()
    print(json.dumps(template, indent=2))


def demonstrate_rule_engine():
    """Demonstrate rule engine functionality."""
    print("\n=== Rule Engine Demo ===\n")
    
    config = ConfigManager()
    engine = RuleEngine()
    
    # Sample metadata
    metadata = {
        'document_type': 'invoice',
        'dates': {
            'document_date': '2023-10-15'
        },
        'entities': {
            'organizations': ['Tech Solutions Inc']
        },
        'amounts': {
            'total': 2500.00
        },
        'tags': ['paid', 'quarterly']
    }
    
    print("Sample document metadata:")
    print(json.dumps(metadata, indent=2))
    print()
    
    # Find matching rule
    rules = config.get_rules()
    matching_rule = engine.find_matching_rule(rules, metadata)
    
    if matching_rule:
        print(f"Matched rule: {matching_rule['name']}")
        print(f"Priority: {matching_rule['priority']}")
        
        # Apply path template
        filename = "INV-2023-001.pdf"
        organized_path = engine.apply_path_template(
            matching_rule['path_template'],
            metadata,
            filename
        )
        print(f"Organized path: {organized_path}\n")
    else:
        print("No matching rule found\n")
    
    # Test multiple metadata samples
    test_cases = [
        {
            'name': 'Receipt',
            'metadata': {
                'document_type': 'receipt',
                'dates': {'document_date': '2023-09-20'},
                'entities': {'organizations': ['Walmart']},
                'amounts': {'total': 45.67}
            },
            'filename': 'receipt_001.jpg'
        },
        {
            'name': 'Photo',
            'metadata': {
                'document_type': 'photo',
                'dates': {'taken_date': '2023-08-15'},
                'tags': ['vacation', 'family']
            },
            'filename': 'IMG_1234.jpg'
        },
        {
            'name': 'Contract',
            'metadata': {
                'document_type': 'contract',
                'dates': {'document_date': '2023-01-10'},
                'entities': {
                    'organizations': ['ABC Corp', 'XYZ Ltd'],
                    'people': ['John Smith', 'Jane Doe']
                }
            },
            'filename': 'contract_final.pdf'
        }
    ]
    
    print("Testing multiple document types:")
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        matching_rule = engine.find_matching_rule(rules, test_case['metadata'])
        
        if matching_rule:
            path = engine.apply_path_template(
                matching_rule['path_template'],
                test_case['metadata'],
                test_case['filename']
            )
            print(f"  Rule: {matching_rule['name']}")
            print(f"  Path: {path}")
        else:
            print("  No matching rule")


def demonstrate_config_persistence():
    """Demonstrate configuration persistence."""
    print("\n=== Configuration Persistence Demo ===\n")
    
    # Create config with custom settings
    config = ConfigManager()
    config.set('api.rate_limit', 25)
    config.set('processing.batch_size', 20)
    config.set('organization.mode', 'move')
    
    # Add a custom rule
    custom_rule = {
        'name': 'Demo Rule',
        'priority': 100,
        'conditions': {'document_type': 'demo'},
        'path_template': 'Demo/{filename}'
    }
    config.add_rule(custom_rule)
    
    # Save configuration
    output_dir = Path(__file__).parent / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "saved_config.json"
    yaml_path = output_dir / "saved_config.yaml"
    
    print("1. Saving configuration:")
    config.save(json_path, format='json')
    config.save(yaml_path, format='yaml')
    print(f"  Saved to: {json_path}")
    print(f"  Saved to: {yaml_path}\n")
    
    # Load saved configuration
    print("2. Loading saved configuration:")
    loaded_config = ConfigManager(config_file=json_path)
    print(f"  Rate limit: {loaded_config.get('api.rate_limit')}")
    print(f"  Batch size: {loaded_config.get('processing.batch_size')}")
    print(f"  Organization mode: {loaded_config.get('organization.mode')}")
    
    rules = loaded_config.get_rules()
    demo_rule = next((r for r in rules if r['name'] == 'Demo Rule'), None)
    print(f"  Custom rule found: {demo_rule is not None}\n")
    
    # Create template
    print("3. Creating configuration template:")
    template_path = output_dir / "config_template.json"
    config.create_template(template_path)
    print(f"  Template saved to: {template_path}")
    
    # Clean up
    import shutil
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    print("Configuration Management System Demo\n")
    
    demonstrate_config_loading()
    demonstrate_rule_management()
    demonstrate_rule_engine()
    demonstrate_config_persistence()
    
    print("\n=== Demo completed successfully! ===")