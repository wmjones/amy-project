"""
Unit tests for preprocessing pipeline module.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.config import PipelineConfig, DEFAULT_CONFIG
from src.preprocessing.enhancer import ImageEnhancer, EnhancementParameters
from src.preprocessing.segmenter import (
    DocumentSegmenter,
    DocumentRegion,
    DocumentRegionType,
)
from src.preprocessing.metadata import MetadataProfile, create_metadata_profile


@pytest.fixture
def test_image():
    """Create a test image."""
    # Create a simple test image with text-like features
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255

    # Add some text-like black rectangles
    cv2.rectangle(image, (50, 50), (550, 100), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 150), (550, 200), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 300), (250, 500), (0, 0, 0), -1)
    cv2.rectangle(image, (350, 300), (550, 500), (0, 0, 0), -1)

    return image


@pytest.fixture
def temp_image_file(test_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, test_image)
        yield f.name
        Path(f.name).unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as dir:
        yield dir


class TestPreprocessingPipeline:
    """Test the PreprocessingPipeline class."""

    def test_initialization_default_config(self):
        """Test pipeline initialization with default config."""
        pipeline = PreprocessingPipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.enhancer, ImageEnhancer)
        assert isinstance(pipeline.segmenter, DocumentSegmenter)

    def test_initialization_custom_config(self):
        """Test pipeline initialization with custom config."""
        custom_config = PipelineConfig(
            enhancement=EnhancementParameters(contrast_factor=2.5),
            segmentation={"min_region_area": 500},
            pipeline={"parallel_processing": False},
            output={"save_enhanced_images": False},
        )

        pipeline = PreprocessingPipeline(config=custom_config)
        assert pipeline.config.enhancement.contrast_factor == 2.5
        assert not pipeline.config.pipeline["parallel_processing"]

    def test_process_document_success(self, temp_image_file, temp_output_dir):
        """Test successful document processing."""
        pipeline = PreprocessingPipeline()
        result = pipeline.process_document(temp_image_file, temp_output_dir)

        assert "metadata" in result
        assert "enhanced_images" in result
        assert "regions" in result
        assert "errors" in result
        assert len(result["errors"]) == 0

        # Check metadata
        metadata = result["metadata"]
        assert isinstance(metadata, MetadataProfile)
        assert len(metadata.processing_steps) > 0
        assert metadata.document_id is not None

    def test_process_document_saves_outputs(self, temp_image_file, temp_output_dir):
        """Test that outputs are saved correctly."""
        config = PipelineConfig(
            enhancement=EnhancementParameters(),
            segmentation=DEFAULT_CONFIG.segmentation,
            pipeline=DEFAULT_CONFIG.pipeline,
            output={
                "save_enhanced_images": True,
                "save_metadata": True,
                "output_format": "png",
                "compression_quality": 95,
                "create_visualization": True,
            },
        )

        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.process_document(temp_image_file, temp_output_dir)

        # Check that files were saved
        output_path = Path(temp_output_dir)
        assert len(list(output_path.glob("*_enhanced_*.png"))) > 0
        assert len(list(output_path.glob("*_metadata.json"))) == 1
        assert len(list(output_path.glob("*_visualization.png"))) == 1

    def test_process_batch(self, temp_image_file, temp_output_dir):
        """Test batch processing of multiple documents."""
        # Create multiple test images
        test_images = [temp_image_file]

        # Create a second test image
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            image = np.ones((400, 300, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (50, 50), (250, 100), (0, 0, 0), -1)
            cv2.imwrite(f.name, image)
            test_images.append(f.name)

        try:
            pipeline = PreprocessingPipeline()
            results = pipeline.process_batch(test_images, temp_output_dir)

            assert len(results) == 2
            for path in test_images:
                assert path in results
                assert (
                    "error" not in results[path]
                    or results[path].get("errors", []) == []
                )
        finally:
            # Clean up second test image
            if len(test_images) > 1:
                Path(test_images[1]).unlink()

    def test_cache_functionality(self, temp_image_file, temp_output_dir):
        """Test caching functionality."""
        config = PipelineConfig(
            enhancement=EnhancementParameters(),
            segmentation=DEFAULT_CONFIG.segmentation,
            pipeline={
                "parallel_processing": True,
                "max_workers": 4,
                "cache_enabled": True,
                "cache_dir": temp_output_dir + "/cache",
                "save_intermediate": False,
                "intermediate_dir": temp_output_dir + "/temp",
            },
            output=DEFAULT_CONFIG.output,
        )

        pipeline = PreprocessingPipeline(config=config)

        # First process
        result1 = pipeline.process_document(temp_image_file, temp_output_dir)

        # Save to cache
        pipeline.save_to_cache(temp_image_file, result1)

        # Check cache
        cached_result = pipeline.check_cache(temp_image_file)
        assert cached_result is not None
        assert "metadata" in cached_result


class TestImageEnhancer:
    """Test the ImageEnhancer class."""

    def test_initialization(self):
        """Test enhancer initialization."""
        enhancer = ImageEnhancer()
        assert enhancer.params is not None
        assert enhancer.params.contrast_factor == 1.5

    def test_enhance_image(self, temp_image_file):
        """Test image enhancement."""
        enhancer = ImageEnhancer()
        results = enhancer.enhance_image(temp_image_file)

        assert "original" in results
        assert "deblurred" in results
        assert "contrast_enhanced" in results
        assert "binarized" in results
        assert "denoised" in results
        assert "sharpened" in results
        assert "combined" in results

        # Check that all results are numpy arrays
        for key, image in results.items():
            assert isinstance(image, np.ndarray)

    def test_save_enhanced_images(self, temp_image_file, temp_output_dir):
        """Test saving enhanced images."""
        enhancer = ImageEnhancer()
        results = enhancer.enhance_image(temp_image_file)

        saved_paths = enhancer.save_enhanced_images(
            results, temp_output_dir, "test_image"
        )

        assert len(saved_paths) == len(results) - 1  # Minus 'error' if any
        for enhancement_type, path in saved_paths.items():
            assert Path(path).exists()


class TestDocumentSegmenter:
    """Test the DocumentSegmenter class."""

    def test_initialization(self):
        """Test segmenter initialization."""
        segmenter = DocumentSegmenter()
        assert segmenter.min_region_area == 1000
        assert segmenter.text_density_threshold == 0.3

    def test_segment_document(self, test_image):
        """Test document segmentation."""
        segmenter = DocumentSegmenter()
        regions = segmenter.segment_document(test_image)

        assert isinstance(regions, list)
        assert len(regions) > 0

        for region in regions:
            assert isinstance(region, DocumentRegion)
            assert region.region_type in DocumentRegionType
            assert region.bbox is not None
            assert region.confidence >= 0 and region.confidence <= 1

    def test_visualize_segmentation(self, test_image):
        """Test segmentation visualization."""
        segmenter = DocumentSegmenter()
        regions = segmenter.segment_document(test_image)

        visualization = segmenter.visualize_segmentation(test_image, regions)
        assert isinstance(visualization, np.ndarray)
        assert visualization.shape == test_image.shape


class TestMetadataProfile:
    """Test metadata profile functionality."""

    def test_create_metadata_profile(self, temp_image_file):
        """Test creating a metadata profile."""
        metadata = create_metadata_profile(temp_image_file)

        assert metadata.document_id is not None
        assert metadata.source_path == str(temp_image_file)
        assert metadata.created_at is not None
        assert metadata.updated_at is not None

    def test_add_processing_step(self, temp_image_file):
        """Test adding processing steps."""
        metadata = create_metadata_profile(temp_image_file)

        metadata.add_processing_step(
            operation="test_operation",
            parameters={"param1": "value1"},
            duration=1.5,
            success=True,
        )

        assert len(metadata.processing_steps) == 1
        step = metadata.processing_steps[0]
        assert step.operation == "test_operation"
        assert step.success is True
        assert step.duration_seconds == 1.5

    def test_serialization(self, temp_image_file, temp_output_dir):
        """Test metadata serialization."""
        metadata = create_metadata_profile(temp_image_file)
        metadata.add_processing_step(operation="test", parameters={}, duration=1.0)

        # To dict
        dict_data = metadata.to_dict()
        assert isinstance(dict_data, dict)
        assert "document_id" in dict_data

        # To JSON
        json_data = metadata.to_json()
        assert isinstance(json_data, str)

        # Save and load
        save_path = Path(temp_output_dir) / "metadata.json"
        metadata.save_to_file(str(save_path))

        loaded_metadata = MetadataProfile.load_from_file(str(save_path))
        assert loaded_metadata.document_id == metadata.document_id
        assert len(loaded_metadata.processing_steps) == len(metadata.processing_steps)


class TestPipelineConfig:
    """Test pipeline configuration functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = DEFAULT_CONFIG
        assert config.enhancement.contrast_factor == 1.5
        assert config.pipeline["parallel_processing"] is True
        assert config.output["save_enhanced_images"] is True

    def test_config_serialization(self, temp_output_dir):
        """Test configuration serialization."""
        config = DEFAULT_CONFIG

        # Save to YAML
        yaml_path = Path(temp_output_dir) / "config.yaml"
        config.save_to_file(str(yaml_path))
        assert yaml_path.exists()

        # Load from YAML
        loaded_config = PipelineConfig.load_from_file(str(yaml_path))
        assert (
            loaded_config.enhancement.contrast_factor
            == config.enhancement.contrast_factor
        )

        # Save to JSON
        json_path = Path(temp_output_dir) / "config.json"
        config.save_to_file(str(json_path))
        assert json_path.exists()

        # Load from JSON
        loaded_config_json = PipelineConfig.load_from_file(str(json_path))
        assert (
            loaded_config_json.enhancement.contrast_factor
            == config.enhancement.contrast_factor
        )
