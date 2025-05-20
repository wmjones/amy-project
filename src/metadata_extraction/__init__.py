"""
Metadata extraction module for document organization.
"""

from .extractor import MetadataExtractor, DocumentMetadata, Entity, DateInfo
from .storage import MetadataStorage

__all__ = [
    "MetadataExtractor",
    "DocumentMetadata",
    "Entity",
    "DateInfo",
    "MetadataStorage",
]
