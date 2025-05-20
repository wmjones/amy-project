"""
Configuration management system for the file organization project.
Handles loading, validation, and merging of configurations from multiple sources.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import argparse
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration from environment variables, files, and command line."""

    def __init__(
        self,
        config_file: Optional[Path] = None,
        cli_args: Optional[argparse.Namespace] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            config_file: Optional path to configuration file
            cli_args: Optional command line arguments
        """
        self.config = self._load_default_config()

        # Load from config file if provided
        if config_file and config_file.exists():
            self._load_from_file(config_file)

        # Override with environment variables
        self._load_from_env()

        # Override with command line arguments
        if cli_args:
            self._load_from_cli(cli_args)

        # Validate configuration
        self._validate_config()

        logger.info("Configuration loaded successfully")

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "api": {
                "anthropic_api_key": None,
                "claude_model": "claude-3-opus-20240229",
                "rate_limit": 10,  # requests per minute
                "max_tokens": 2000,
                "timeout": 30,
                "max_retries": 3,
                "initial_backoff": 1.0,
            },
            "dropbox": {
                "app_key": None,
                "app_secret": None,
                "access_token": None,
                "source_folder": "/file-organizer-uploads",
                "download_batch_size": 10,
            },
            "processing": {
                "batch_size": 10,
                "max_workers": 4,
                "file_types": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tiff",
                    ".bmp",
                    ".pdf",
                    ".docx",
                    ".txt",
                ],
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "queue_priority": "fifo",  # 'fifo', 'size', 'type'
                "enable_ocr": True,
                "ocr_languages": ["eng"],
            },
            "organization": {
                "mode": "copy",  # 'copy' or 'move'
                "base_directory": "./organized",
                "conflict_resolution": "rename",  # 'skip', 'rename', 'overwrite'
                "preserve_structure": False,
                "rules": self._load_default_rules(),
            },
            "metadata": {
                "storage_backend": "sqlite",  # 'sqlite' or 'json'
                "db_path": "./metadata.db",
                "json_path": "./metadata.json",
                "index_fields": [
                    "document_type",
                    "dates.document_date",
                    "entities.people",
                    "entities.organizations",
                ],
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/organizer.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "ui": {
                "show_progress": True,
                "update_interval": 1,  # seconds
                "color_output": True,
            },
        }

    def _load_default_rules(self) -> List[Dict[str, Any]]:
        """Load default organization rules."""
        return [
            {
                "name": "Documents by type",
                "priority": 1,
                "conditions": {"document_type": "*"},
                "path_template": "{document_type}/{filename}",
                "enabled": True,
            },
            {
                "name": "Documents by date",
                "priority": 2,
                "conditions": {
                    "dates.document_date": "*",
                    "dates.document_date": {"$gte": "2020-01-01"},
                },
                "path_template": "{dates.document_date|%Y}/{dates.document_date|%B}/{filename}",
                "enabled": True,
            },
            {
                "name": "Receipts by store",
                "priority": 3,
                "conditions": {
                    "document_type": "receipt",
                    "entities.organizations": "*",
                },
                "path_template": "Receipts/{entities.organizations[0]}/{dates.document_date|%Y-%m}/{filename}",
                "enabled": True,
            },
            {
                "name": "Invoices by client",
                "priority": 4,
                "conditions": {
                    "document_type": "invoice",
                    "entities.organizations": "*",
                },
                "path_template": "Invoices/{entities.organizations[0]}/{dates.document_date|%Y}/{filename}",
                "enabled": True,
            },
            {
                "name": "Photos by date",
                "priority": 5,
                "conditions": {"document_type": "photo", "dates.taken_date": "*"},
                "path_template": "Photos/{dates.taken_date|%Y}/{dates.taken_date|%Y-%m-%d}/{filename}",
                "enabled": True,
            },
        ]

    def _load_from_file(self, config_file: Path):
        """Load configuration from file."""
        logger.info(f"Loading configuration from {config_file}")

        try:
            with open(config_file, "r") as f:
                if config_file.suffix == ".json":
                    file_config = json.load(f)
                elif config_file.suffix in (".yaml", ".yml"):
                    file_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file}")

            # Deep merge with default config
            self._deep_merge(self.config, file_config)

        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Special handling for API keys
        api_keys = {
            "ANTHROPIC_API_KEY": ["api", "anthropic_api_key"],
            "DROPBOX_APP_KEY": ["dropbox", "app_key"],
            "DROPBOX_APP_SECRET": ["dropbox", "app_secret"],
            "DROPBOX_ACCESS_TOKEN": ["dropbox", "access_token"],
        }

        for env_var, config_path in api_keys.items():
            if env_var in os.environ:
                self._set_nested_config(self.config, config_path, os.environ[env_var])

        # Load other environment variables with FILE_ORGANIZER_ prefix
        for key, value in os.environ.items():
            if key.startswith("FILE_ORGANIZER_"):
                config_path = key[15:].lower().split("__")
                self._set_nested_config(self.config, config_path, value)

    def _load_from_cli(self, cli_args: argparse.Namespace):
        """Load configuration from command line arguments."""
        # Map CLI arguments to configuration paths
        cli_mappings = {
            "source_dir": ["organization", "source_directory"],
            "output_dir": ["organization", "base_directory"],
            "batch_size": ["processing", "batch_size"],
            "mode": ["organization", "mode"],
            "max_workers": ["processing", "max_workers"],
            "log_level": ["logging", "level"],
            "config_file": None,  # Special case - already handled
        }

        for arg_name, config_path in cli_mappings.items():
            if hasattr(cli_args, arg_name) and getattr(cli_args, arg_name) is not None:
                if config_path:  # Skip special cases
                    self._set_nested_config(
                        self.config, config_path, getattr(cli_args, arg_name)
                    )

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge update dictionary into base dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _set_nested_config(
        self, config_dict: Dict[str, Any], path: List[str], value: Any
    ):
        """Set a value in a nested dictionary using a path."""
        # Convert value to appropriate type if it's a string
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                value = float(value)
            elif value.startswith("[") and value.endswith("]"):
                # Parse list
                try:
                    value = json.loads(value)
                except:
                    pass

        # Set value in nested dictionary
        current = config_dict
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = value

    def _validate_config(self):
        """Validate configuration values."""
        errors = []

        # API configuration
        if not self.config["api"]["anthropic_api_key"]:
            logger.warning("No Anthropic API key configured")

        # Dropbox configuration
        dropbox_config = self.config["dropbox"]
        if dropbox_config["app_key"] or dropbox_config["app_secret"]:
            if not all([dropbox_config["app_key"], dropbox_config["app_secret"]]):
                errors.append(
                    "Incomplete Dropbox configuration: both app_key and app_secret required"
                )

        # Processing configuration
        if self.config["processing"]["batch_size"] < 1:
            errors.append("batch_size must be >= 1")

        if self.config["processing"]["max_workers"] < 1:
            errors.append("max_workers must be >= 1")

        # Organization configuration
        if self.config["organization"]["mode"] not in ["copy", "move"]:
            errors.append("organization mode must be 'copy' or 'move'")

        if self.config["organization"]["conflict_resolution"] not in [
            "skip",
            "rename",
            "overwrite",
        ]:
            errors.append(
                "conflict_resolution must be 'skip', 'rename', or 'overwrite'"
            )

        # Metadata configuration
        if self.config["metadata"]["storage_backend"] not in ["sqlite", "json"]:
            errors.append("metadata storage_backend must be 'sqlite' or 'json'")

        # Logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config["logging"]["level"] not in valid_log_levels:
            errors.append(f"logging level must be one of {valid_log_levels}")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            path: Configuration path (e.g., 'api.rate_limit')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        parts = path.split(".")
        current = self.config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            path: Configuration path (e.g., 'api.rate_limit')
            value: Value to set
        """
        parts = path.split(".")
        current = self.config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def save(self, filepath: Path, format: str = "json"):
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        logger.info(f"Saving configuration to {filepath}")

        with open(filepath, "w") as f:
            if format == "json":
                json.dump(self.config, f, indent=2)
            elif format in ("yaml", "yml"):
                yaml.safe_dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def create_template(self, filepath: Path, format: str = "json"):
        """Create a configuration template file."""
        template_config = deepcopy(self.config)

        # Add comments/descriptions
        if format == "json":
            # JSON doesn't support comments, so we'll add _comment fields
            template_config["_comment"] = "File Organizer Configuration Template"
            template_config["api"][
                "_comment"
            ] = "API configuration for Claude/Anthropic"
            template_config["dropbox"]["_comment"] = "Dropbox integration settings"
            template_config["processing"]["_comment"] = "File processing settings"
            template_config["organization"][
                "_comment"
            ] = "File organization rules and settings"
            template_config["metadata"]["_comment"] = "Metadata storage configuration"
            template_config["logging"]["_comment"] = "Logging configuration"
            template_config["ui"] = template_config.get("ui", {})
            template_config["ui"]["_comment"] = "User interface settings"

            # Save the template with comments
            with open(filepath, "w") as f:
                json.dump(template_config, f, indent=2)
        else:
            self.save(filepath, format)

        logger.info(f"Configuration template created at {filepath}")

    def get_rule_template(self) -> Dict[str, Any]:
        """Get a template for creating custom organization rules."""
        return {
            "name": "Custom Rule",
            "priority": 10,
            "conditions": {
                "document_type": "example",
                "metadata_field": {"$exists": True, "$gte": "value"},
            },
            "path_template": "{document_type}/{metadata_field}/{filename}",
            "enabled": True,
            "description": "Description of what this rule does",
        }

    def add_rule(self, rule: Dict[str, Any]):
        """Add a custom organization rule."""
        if "name" not in rule:
            raise ValueError("Rule must have a name")

        if "conditions" not in rule or "path_template" not in rule:
            raise ValueError("Rule must have conditions and path_template")

        # Add default values
        rule.setdefault("priority", 100)
        rule.setdefault("enabled", True)

        self.config["organization"]["rules"].append(rule)

        # Sort rules by priority
        self.config["organization"]["rules"].sort(key=lambda r: r.get("priority", 100))

        logger.info(f"Added organization rule: {rule['name']}")

    def get_rules(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get organization rules."""
        rules = self.config["organization"]["rules"]

        if enabled_only:
            rules = [r for r in rules if r.get("enabled", True)]

        return rules

    def validate_rule(self, rule: Dict[str, Any]) -> List[str]:
        """Validate an organization rule and return errors."""
        errors = []

        if "name" not in rule:
            errors.append("Rule must have a name")

        if "conditions" not in rule:
            errors.append("Rule must have conditions")
        elif not isinstance(rule["conditions"], dict):
            errors.append("Rule conditions must be a dictionary")

        if "path_template" not in rule:
            errors.append("Rule must have a path_template")
        elif not isinstance(rule["path_template"], str):
            errors.append("Rule path_template must be a string")

        # Validate path template placeholders
        if "path_template" in rule:
            import re

            placeholders = re.findall(r"\{([^}]+)\}", rule["path_template"])
            for placeholder in placeholders:
                if "|" in placeholder:
                    # Format specified
                    field, _ = placeholder.split("|", 1)
                else:
                    field = placeholder

                # Check if field is referenced in conditions
                if not self._is_field_in_conditions(field, rule.get("conditions", {})):
                    errors.append(
                        f"Path template references '{field}' which is not in conditions"
                    )

        return errors

    def _is_field_in_conditions(self, field: str, conditions: Dict[str, Any]) -> bool:
        """Check if a field is referenced in conditions."""
        # 'filename' is a special built-in field that's always available
        if field == "filename":
            return True

        for key, value in conditions.items():
            if key == field:
                return True
            if isinstance(value, dict):
                if self._is_field_in_conditions(field, value):
                    return True
        return False
