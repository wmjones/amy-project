"""
Integration module for connecting different components of the file organization system.
"""

from .metadata_integration import MetadataIntegrationBridge, MetadataConflict

__all__ = ["MetadataIntegrationBridge", "MetadataConflict"]
