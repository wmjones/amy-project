"""
Main preprocessing pipeline orchestrator.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import cv2
import numpy as np

from .enhancer import ImageEnhancer
from .segmenter import DocumentSegmenter, DocumentRegion
from .metadata import MetadataProfile, create_metadata_profile, calculate_quality_metrics
from .config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline for document images."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the preprocessing pipeline.
        
        Args:
            config: Pipeline configuration (uses default if not provided)
        """
        self.config = config or load_config()
        self.enhancer = ImageEnhancer(self.config.enhancement)
        self.segmenter = DocumentSegmenter(
            min_region_area=self.config.segmentation['min_region_area'],
            text_density_threshold=self.config.segmentation['text_density_threshold'],
            line_detection_threshold=self.config.segmentation['line_detection_threshold']
        )
        
        # Setup cache directory if enabled
        if self.config.pipeline['cache_enabled']:
            self.cache_dir = Path(self.config.pipeline['cache_dir'])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup intermediate directory if saving intermediates
        if self.config.pipeline['save_intermediate']:
            self.intermediate_dir = Path(self.config.pipeline['intermediate_dir'])
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    def process_document(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single document through the complete pipeline.
        
        Args:
            image_path: Path to the input document image
            output_dir: Directory to save processed outputs
            
        Returns:
            Dictionary containing:
                - metadata: Document metadata profile
                - enhanced_images: Paths to enhanced image variants
                - regions: List of detected document regions
                - errors: Any errors encountered during processing
        """
        start_time = time.time()
        logger.info(f"Processing document: {image_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata profile
        metadata = create_metadata_profile(image_path)
        
        result = {
            'metadata': metadata,
            'enhanced_images': {},
            'regions': [],
            'errors': []
        }
        
        try:
            # Step 1: Enhance image
            enhancement_start = time.time()
            enhanced_images = self.enhancer.enhance_image(image_path)
            enhancement_time = time.time() - enhancement_start
            
            metadata.add_processing_step(
                operation='image_enhancement',
                parameters=self.config.enhancement.__dict__,
                duration=enhancement_time,
                success=True,
                output_info={'variants': list(enhanced_images.keys())}
            )
            
            # Update image characteristics
            original_image = enhanced_images['original']
            metadata.image_characteristics.height, metadata.image_characteristics.width = original_image.shape[:2]
            metadata.image_characteristics.channels = original_image.shape[2] if len(original_image.shape) > 2 else 1
            
            # Save enhanced images if configured
            if self.config.output['save_enhanced_images']:
                save_start = time.time()
                base_filename = Path(image_path).stem
                saved_paths = self.enhancer.save_enhanced_images(
                    enhanced_images,
                    output_dir,
                    base_filename
                )
                result['enhanced_images'] = saved_paths
                metadata.enhancement_variants = saved_paths
                
                metadata.add_processing_step(
                    operation='save_enhanced_images',
                    parameters={'output_dir': str(output_dir)},
                    duration=time.time() - save_start,
                    success=True,
                    output_info={'saved_files': len(saved_paths)}
                )
            
            # Step 2: Segment document
            segmentation_start = time.time()
            regions = self.segmenter.segment_document(enhanced_images['combined'])
            segmentation_time = time.time() - segmentation_start
            
            result['regions'] = regions
            metadata.document_regions = [self._region_to_dict(region) for region in regions]
            
            metadata.add_processing_step(
                operation='document_segmentation',
                parameters=self.config.segmentation,
                duration=segmentation_time,
                success=True,
                output_info={'regions_found': len(regions)}
            )
            
            # Step 3: Calculate quality metrics
            quality_start = time.time()
            quality_metrics = calculate_quality_metrics(enhanced_images['combined'])
            metadata.quality_metrics = quality_metrics
            
            metadata.add_processing_step(
                operation='quality_assessment',
                parameters={},
                duration=time.time() - quality_start,
                success=True,
                output_info={'overall_quality': quality_metrics.overall_quality}
            )
            
            # Step 4: Create visualization if configured
            if self.config.output['create_visualization']:
                vis_start = time.time()
                visualization = self.segmenter.visualize_segmentation(
                    enhanced_images['original'],
                    regions
                )
                vis_path = output_dir / f"{Path(image_path).stem}_visualization.png"
                cv2.imwrite(str(vis_path), visualization)
                
                metadata.add_processing_step(
                    operation='create_visualization',
                    parameters={},
                    duration=time.time() - vis_start,
                    success=True,
                    output_info={'visualization_path': str(vis_path)}
                )
            
            # Save metadata if configured
            if self.config.output['save_metadata']:
                metadata_path = output_dir / f"{Path(image_path).stem}_metadata.json"
                metadata.save_to_file(str(metadata_path))
            
            # Add final processing step
            total_time = time.time() - start_time
            metadata.add_processing_step(
                operation='complete_pipeline',
                parameters={'config': self.config.to_dict()},
                duration=total_time,
                success=True,
                output_info={
                    'total_steps': len(metadata.processing_steps),
                    'output_dir': str(output_dir)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing document {image_path}: {e}")
            result['errors'].append(str(e))
            metadata.add_processing_step(
                operation='pipeline_error',
                parameters={},
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        
        result['metadata'] = metadata
        return result
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Process multiple documents in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Base directory for outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping image paths to their processing results
        """
        results = {}
        total_files = len(image_paths)
        
        if self.config.pipeline['parallel_processing']:
            # Process in parallel
            max_workers = self.config.pipeline['max_workers']
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self.process_document,
                        path,
                        Path(output_dir) / Path(path).stem
                    ): path
                    for path in image_paths
                }
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_path)):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        results[path] = {'error': str(e)}
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files)
        else:
            # Process sequentially
            for i, path in enumerate(image_paths):
                try:
                    result = self.process_document(
                        path,
                        Path(output_dir) / Path(path).stem
                    )
                    results[path] = result
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results[path] = {'error': str(e)}
                
                if progress_callback:
                    progress_callback(i + 1, total_files)
        
        return results
    
    def _region_to_dict(self, region: DocumentRegion) -> Dict[str, Any]:
        """Convert a DocumentRegion to a dictionary.
        
        Args:
            region: Document region object
            
        Returns:
            Dictionary representation
        """
        return {
            'type': region.region_type.value,
            'bbox': region.bbox,
            'confidence': region.confidence,
            'text_density': region.text_density,
            'is_vertical': region.is_vertical,
            'area': region.area,
            'center': region.center,
            'metadata': region.metadata
        }
    
    def get_cache_key(self, image_path: str) -> str:
        """Generate cache key for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Include config in hash for cache invalidation
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        content = f"{image_path}:{config_str}"
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def check_cache(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Check if processed results exist in cache.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Cached results if available, None otherwise
        """
        if not self.config.pipeline['cache_enabled']:
            return None
        
        cache_key = self.get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Convert back to proper objects
                if 'metadata' in cached_data:
                    cached_data['metadata'] = MetadataProfile.from_dict(
                        cached_data['metadata']
                    )
                
                logger.info(f"Found cached results for {image_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {image_path}: {e}")
        
        return None
    
    def save_to_cache(self, image_path: str, result: Dict[str, Any]):
        """Save processing results to cache.
        
        Args:
            image_path: Path to the image
            result: Processing results
        """
        if not self.config.pipeline['cache_enabled']:
            return
        
        cache_key = self.get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Convert objects to serializable format
            cache_data = result.copy()
            if 'metadata' in cache_data and hasattr(cache_data['metadata'], 'to_dict'):
                cache_data['metadata'] = cache_data['metadata'].to_dict()
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.debug(f"Saved results to cache for {image_path}")
        except Exception as e:
            logger.warning(f"Failed to cache results for {image_path}: {e}")