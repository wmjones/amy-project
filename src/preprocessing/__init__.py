"""Preprocessing pipeline for document images.

This module provides image enhancement and document structure analysis
for improved OCR and AI processing."""

from .pipeline import PreprocessingPipeline
from .enhancer import ImageEnhancer, EnhancementParameters
from .segmenter import DocumentSegmenter, DocumentRegion, DocumentRegionType
from .metadata import MetadataProfile, create_metadata_profile
from .config import PipelineConfig, load_config, DEFAULT_CONFIG, FAST_CONFIG, HIGH_QUALITY_CONFIG
from .ocr_enhanced import (
    EnhancedOCRProcessor, 
    EnhancedOCRResult,
    CharacterConfidence,
    WordConfidence,
    LineData,
    ProblematicRegion
)

__all__ = [
    'PreprocessingPipeline',
    'ImageEnhancer',
    'EnhancementParameters',
    'DocumentSegmenter',
    'DocumentRegion',
    'DocumentRegionType',
    'MetadataProfile',
    'create_metadata_profile',
    'PipelineConfig',
    'load_config',
    'DEFAULT_CONFIG',
    'FAST_CONFIG',
    'HIGH_QUALITY_CONFIG',
    'EnhancedOCRProcessor',
    'EnhancedOCRResult',
    'CharacterConfidence',
    'WordConfidence',
    'LineData',
    'ProblematicRegion'
]