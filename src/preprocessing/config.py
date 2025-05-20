"""
Configuration system for preprocessing pipeline.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, asdict

from .enhancer import EnhancementParameters

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    
    # Image enhancement settings
    enhancement: EnhancementParameters
    
    # Document segmentation settings
    segmentation: Dict[str, Any]
    
    # Pipeline processing settings
    pipeline: Dict[str, Any]
    
    # Output settings
    output: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PipelineConfig instance
        """
        enhancement = EnhancementParameters(**config_dict.get('enhancement', {}))
        
        return cls(
            enhancement=enhancement,
            segmentation=config_dict.get('segmentation', {
                'min_region_area': 1000,
                'text_density_threshold': 0.3,
                'line_detection_threshold': 150
            }),
            pipeline=config_dict.get('pipeline', {
                'parallel_processing': True,
                'max_workers': 4,
                'cache_enabled': True,
                'cache_dir': 'workspace/cache/preprocessing',
                'save_intermediate': False,
                'intermediate_dir': 'workspace/temp/preprocessing'
            }),
            output=config_dict.get('output', {
                'save_enhanced_images': True,
                'save_metadata': True,
                'output_format': 'png',
                'compression_quality': 95,
                'create_visualization': False
            })
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'enhancement': asdict(self.enhancement),
            'segmentation': self.segmentation,
            'pipeline': self.pipeline,
            'output': self.output
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to file.
        
        Args:
            file_path: Path to save the configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    if hasattr(obj, 'tolist'):  # numpy array
                        return obj.tolist()
                    return super().default(obj)
            
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved configuration to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PipelineConfig':
        """Load configuration from file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            PipelineConfig instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Default configuration presets
DEFAULT_CONFIG = PipelineConfig(
    enhancement=EnhancementParameters(),
    segmentation={
        'min_region_area': 1000,
        'text_density_threshold': 0.3,
        'line_detection_threshold': 150
    },
    pipeline={
        'parallel_processing': True,
        'max_workers': 4,
        'cache_enabled': True,
        'cache_dir': 'workspace/cache/preprocessing',
        'save_intermediate': False,
        'intermediate_dir': 'workspace/temp/preprocessing'
    },
    output={
        'save_enhanced_images': True,
        'save_metadata': True,
        'output_format': 'png',
        'compression_quality': 95,
        'create_visualization': False
    }
)

# High quality scanning preset
HIGH_QUALITY_CONFIG = PipelineConfig(
    enhancement=EnhancementParameters(
        deblur_strength=1.5,
        contrast_factor=2.0,
        brightness_factor=1.1,
        denoising_strength=5,
        sharpening_amount=1.5
    ),
    segmentation={
        'min_region_area': 500,
        'text_density_threshold': 0.25,
        'line_detection_threshold': 100
    },
    pipeline={
        'parallel_processing': True,
        'max_workers': 2,  # Less workers for more intensive processing
        'cache_enabled': True,
        'cache_dir': 'workspace/cache/preprocessing',
        'save_intermediate': True,
        'intermediate_dir': 'workspace/temp/preprocessing'
    },
    output={
        'save_enhanced_images': True,
        'save_metadata': True,
        'output_format': 'png',
        'compression_quality': 100,
        'create_visualization': True
    }
)

# Fast processing preset
FAST_CONFIG = PipelineConfig(
    enhancement=EnhancementParameters(
        deblur_strength=0.5,
        contrast_factor=1.2,
        brightness_factor=1.0,
        denoising_strength=1,
        sharpening_amount=0.8
    ),
    segmentation={
        'min_region_area': 2000,
        'text_density_threshold': 0.4,
        'line_detection_threshold': 200
    },
    pipeline={
        'parallel_processing': True,
        'max_workers': 8,
        'cache_enabled': True,
        'cache_dir': 'workspace/cache/preprocessing',
        'save_intermediate': False,
        'intermediate_dir': 'workspace/temp/preprocessing'
    },
    output={
        'save_enhanced_images': False,
        'save_metadata': True,
        'output_format': 'jpg',
        'compression_quality': 85,
        'create_visualization': False
    }
)


def load_config(config_path: Optional[str] = None, preset: Optional[str] = None) -> PipelineConfig:
    """Load configuration from file or preset.
    
    Args:
        config_path: Path to configuration file
        preset: Preset name ('default', 'high_quality', 'fast')
        
    Returns:
        PipelineConfig instance
    """
    if config_path:
        return PipelineConfig.load_from_file(config_path)
    
    presets = {
        'default': DEFAULT_CONFIG,
        'high_quality': HIGH_QUALITY_CONFIG,
        'fast': FAST_CONFIG
    }
    
    if preset and preset in presets:
        return presets[preset]
    
    return DEFAULT_CONFIG


def create_config_template(output_path: str):
    """Create a configuration template file.
    
    Args:
        output_path: Path to save the template
    """
    template_config = DEFAULT_CONFIG.to_dict()
    
    # Add comments to the template
    template_with_comments = {
        '_comment': 'Preprocessing pipeline configuration template',
        'enhancement': {
            '_comment': 'Image enhancement parameters',
            **template_config['enhancement']
        },
        'segmentation': {
            '_comment': 'Document segmentation parameters',
            **template_config['segmentation']
        },
        'pipeline': {
            '_comment': 'Pipeline processing settings',
            **template_config['pipeline']
        },
        'output': {
            '_comment': 'Output settings',
            **template_config['output']
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(template_with_comments, f, default_flow_style=False)
    else:
        with open(output_path, 'w') as f:
            json.dump(template_with_comments, f, indent=2)
    
    logger.info(f"Created configuration template at {output_path}")