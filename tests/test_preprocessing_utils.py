"""
Unit tests for preprocessing utility functions.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.preprocessing.enhancer import EnhancementParameters
from src.preprocessing.segmenter import DocumentRegionType, DocumentRegion
from src.preprocessing.metadata import (
    ProcessingStep,
    ImageCharacteristics,
    QualityMetrics,
    calculate_quality_metrics
)
from src.preprocessing.config import load_config, create_config_template


class TestEnhancementParameters:
    """Test enhancement parameters."""
    
    def test_default_parameters(self):
        """Test default enhancement parameters."""
        params = EnhancementParameters()
        assert params.deblur_strength == 1.0
        assert params.contrast_factor == 1.5
        assert params.brightness_factor == 1.0
        assert params.binarization_threshold == 127
        assert params.adaptive_binarization is True
        assert params.denoising_strength == 3
        assert params.sharpening_amount == 1.0
    
    def test_custom_parameters(self):
        """Test custom enhancement parameters."""
        params = EnhancementParameters(
            deblur_strength=2.0,
            contrast_factor=2.5,
            adaptive_binarization=False
        )
        assert params.deblur_strength == 2.0
        assert params.contrast_factor == 2.5
        assert params.adaptive_binarization is False


class TestDocumentRegion:
    """Test document region functionality."""
    
    def test_region_creation(self):
        """Test creating a document region."""
        region = DocumentRegion(
            region_type=DocumentRegionType.PARAGRAPH,
            bbox=(10, 20, 100, 50),
            confidence=0.95,
            text_density=0.8
        )
        
        assert region.region_type == DocumentRegionType.PARAGRAPH
        assert region.bbox == (10, 20, 100, 50)
        assert region.confidence == 0.95
        assert region.text_density == 0.8
    
    def test_region_properties(self):
        """Test document region properties."""
        region = DocumentRegion(
            region_type=DocumentRegionType.TABLE,
            bbox=(0, 0, 100, 200),
            confidence=0.9
        )
        
        assert region.area == 20000  # 100 * 200
        assert region.center == (50, 100)  # (0+100)/2, (0+200)/2


class TestProcessingStep:
    """Test processing step functionality."""
    
    def test_step_creation(self):
        """Test creating a processing step."""
        step = ProcessingStep(
            operation='test_operation',
            timestamp='2023-01-01T00:00:00',
            parameters={'param': 'value'},
            duration_seconds=1.5,
            success=True
        )
        
        assert step.operation == 'test_operation'
        assert step.success is True
        assert step.duration_seconds == 1.5
        assert step.error_message is None
    
    def test_step_with_error(self):
        """Test processing step with error."""
        step = ProcessingStep(
            operation='failed_operation',
            timestamp='2023-01-01T00:00:00',
            parameters={},
            duration_seconds=0.5,
            success=False,
            error_message='Test error message'
        )
        
        assert step.success is False
        assert step.error_message == 'Test error message'


class TestImageCharacteristics:
    """Test image characteristics functionality."""
    
    def test_characteristics_creation(self):
        """Test creating image characteristics."""
        chars = ImageCharacteristics(
            width=800,
            height=600,
            channels=3,
            format='png',
            dpi=300
        )
        
        assert chars.width == 800
        assert chars.height == 600
        assert chars.resolution == (800, 600)
        assert chars.aspect_ratio == 800/600
        assert chars.dpi == 300


class TestQualityMetrics:
    """Test quality metrics functionality."""
    
    def test_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            sharpness_score=0.9,
            contrast_score=0.85,
            brightness_score=0.8,
            text_clarity_score=0.88,
            noise_level=0.1,
            skew_angle=0.5,
            overall_quality=0.87
        )
        
        assert metrics.sharpness_score == 0.9
        assert metrics.overall_quality == 0.87
    
    def test_calculate_quality_metrics(self):
        """Test calculating quality metrics (placeholder test)."""
        # Create a simple test image
        image = np.ones((100, 100), dtype=np.uint8) * 128
        
        metrics = calculate_quality_metrics(image)
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.overall_quality <= 1


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config()
        assert config is not None
        assert config.enhancement.contrast_factor == 1.5
    
    def test_load_config_preset(self):
        """Test loading configuration presets."""
        # Test fast preset
        fast_config = load_config(preset='fast')
        assert fast_config.enhancement.deblur_strength == 0.5
        assert fast_config.pipeline['max_workers'] == 8
        
        # Test high quality preset
        hq_config = load_config(preset='high_quality')
        assert hq_config.enhancement.contrast_factor == 2.0
        assert hq_config.output['compression_quality'] == 100
    
    def test_create_config_template(self, temp_output_dir):
        """Test creating a configuration template."""
        yaml_path = Path(temp_output_dir) / 'template.yaml'
        create_config_template(str(yaml_path))
        
        assert yaml_path.exists()
        
        # Test JSON template
        json_path = Path(temp_output_dir) / 'template.json'
        create_config_template(str(json_path))
        
        assert json_path.exists()


@pytest.mark.parametrize("region_type,expected_str", [
    (DocumentRegionType.HEADER, "header"),
    (DocumentRegionType.PARAGRAPH, "paragraph"),
    (DocumentRegionType.TABLE, "table"),
    (DocumentRegionType.FIGURE, "figure"),
    (DocumentRegionType.FOOTER, "footer"),
    (DocumentRegionType.CAPTION, "caption"),
    (DocumentRegionType.LIST, "list"),
    (DocumentRegionType.UNKNOWN, "unknown")
])
def test_document_region_types(region_type, expected_str):
    """Test document region type enum values."""
    assert region_type.value == expected_str
