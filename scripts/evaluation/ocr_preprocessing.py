#!/usr/bin/env python3
"""OCR Pre-processing Optimization Framework for Hansman Syracuse Collection."""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from skimage.transform import rotate
from sklearn.linear_model import LinearRegression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageQualityMetrics:
    """Metrics for assessing document image quality."""
    contrast: float
    brightness: float
    sharpness: float
    noise_level: float
    skew_angle: float
    has_borders: bool
    resolution: Tuple[int, int]
    blur_score: float
    
class DocumentPreprocessor:
    """Pre-processing pipeline for historical document images."""
    
    def __init__(self):
        self.techniques = {
            'deskew': self.deskew_image,
            'denoise': self.denoise_image,
            'enhance_contrast': self.enhance_contrast,
            'remove_borders': self.remove_borders,
            'binarize': self.binarize_image,
            'sharpen': self.sharpen_image,
            'resize': self.resize_image,
            'morphological_cleaning': self.morphological_cleaning
        }
        self.quality_thresholds = {
            'min_contrast': 0.3,
            'min_sharpness': 0.5,
            'max_skew': 5.0,  # degrees
            'min_resolution': (800, 600)
        }
    
    def analyze_image_quality(self, image_path: Path) -> ImageQualityMetrics:
        """Analyze document image quality metrics."""
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast
        contrast = gray.std()
        
        # Calculate brightness
        brightness = gray.mean()
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Estimate noise level
        noise_level = self._estimate_noise(gray)
        
        # Detect skew angle
        skew_angle = self._detect_skew(gray)
        
        # Check for borders
        has_borders = self._detect_borders(gray)
        
        # Get resolution
        resolution = (image.shape[1], image.shape[0])
        
        # Calculate blur score
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return ImageQualityMetrics(
            contrast=contrast,
            brightness=brightness,
            sharpness=sharpness,
            noise_level=noise_level,
            skew_angle=skew_angle,
            has_borders=has_borders,
            resolution=resolution,
            blur_score=blur_score
        )
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image."""
        # Use median absolute deviation
        median = np.median(gray_image)
        mad = np.median(np.abs(gray_image - median))
        return mad
    
    def _detect_skew(self, gray_image: np.ndarray) -> float:
        """Detect skew angle of document."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:  # Filter reasonable angles
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0
    
    def _detect_borders(self, gray_image: np.ndarray) -> bool:
        """Detect if image has dark borders."""
        h, w = gray_image.shape
        
        # Check edges
        top_edge = gray_image[0:10, :].mean()
        bottom_edge = gray_image[h-10:h, :].mean()
        left_edge = gray_image[:, 0:10].mean()
        right_edge = gray_image[:, w-10:w].mean()
        
        # Dark borders typically have low intensity
        border_threshold = 50
        return any(edge < border_threshold for edge in [top_edge, bottom_edge, left_edge, right_edge])
    
    def deskew_image(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Deskew document image."""
        if angle is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            angle = self._detect_skew(gray)
        
        if abs(angle) > 0.5:  # Only deskew if angle is significant
            logger.info(f"Deskewing by {angle:.2f} degrees")
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from document image."""
        if len(image.shape) == 3:
            # For color images
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # For grayscale
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using adaptive histogram equalization."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove dark borders from scanned documents."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (usually the document)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop image
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image
    
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image for better OCR."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use adaptive thresholding for better results on historical documents
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen blurry text."""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def resize_image(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Resize image to optimal DPI for OCR."""
        # Assuming standard document size (8.5x11 inches)
        standard_width = int(8.5 * target_dpi)
        standard_height = int(11 * target_dpi)
        
        h, w = image.shape[:2]
        
        # Calculate scaling factors
        scale_w = standard_width / w
        scale_h = standard_height / h
        scale = min(scale_w, scale_h)
        
        if scale > 1.0:  # Only upscale if necessary
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def morphological_cleaning(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up text."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Close small gaps in text
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def create_preprocessing_pipeline(self, metrics: ImageQualityMetrics) -> List[str]:
        """Create optimal preprocessing pipeline based on image metrics."""
        pipeline = []
        
        # Always start with denoising if noise is high
        if metrics.noise_level > 5:
            pipeline.append('denoise')
        
        # Deskew if needed
        if abs(metrics.skew_angle) > self.quality_thresholds['max_skew']:
            pipeline.append('deskew')
        
        # Remove borders if detected
        if metrics.has_borders:
            pipeline.append('remove_borders')
        
        # Enhance contrast if low
        if metrics.contrast < self.quality_thresholds['min_contrast']:
            pipeline.append('enhance_contrast')
        
        # Sharpen if blurry
        if metrics.sharpness < self.quality_thresholds['min_sharpness']:
            pipeline.append('sharpen')
        
        # Resize if resolution is too low
        if (metrics.resolution[0] < self.quality_thresholds['min_resolution'][0] or
            metrics.resolution[1] < self.quality_thresholds['min_resolution'][1]):
            pipeline.append('resize')
        
        # Always end with morphological cleaning and binarization
        pipeline.extend(['morphological_cleaning', 'binarize'])
        
        return pipeline
    
    def process_image(self, image_path: Path, save_intermediate: bool = False) -> Dict:
        """Process single image with optimal pipeline."""
        logger.info(f"Processing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        original = image.copy()
        
        # Analyze quality
        metrics = self.analyze_image_quality(image_path)
        
        # Create pipeline
        pipeline = self.create_preprocessing_pipeline(metrics)
        logger.info(f"Pipeline: {' -> '.join(pipeline)}")
        
        # Apply pipeline
        results = {'original': original, 'steps': {}}
        
        for step in pipeline:
            if step in self.techniques:
                image = self.techniques[step](image)
                results['steps'][step] = image.copy()
                
                if save_intermediate:
                    step_path = image_path.parent / f"{image_path.stem}_{step}{image_path.suffix}"
                    cv2.imwrite(str(step_path), image)
        
        results['final'] = image
        results['metrics'] = metrics
        results['pipeline'] = pipeline
        
        return results
    
    def batch_process(self, input_folder: Path, output_folder: Path, 
                     sample_size: Optional[int] = None) -> Dict:
        """Process multiple images and collect statistics."""
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = list(input_folder.glob("*.JPG")) + list(input_folder.glob("*.jpg"))
        
        if sample_size:
            image_files = image_files[:sample_size]
        
        logger.info(f"Processing {len(image_files)} images")
        
        statistics = {
            'total_files': len(image_files),
            'successful': 0,
            'failed': [],
            'pipeline_usage': {},
            'average_metrics': {}
        }
        
        all_metrics = []
        
        for image_path in image_files:
            try:
                # Process image
                results = self.process_image(image_path)
                
                # Save processed image
                output_path = output_folder / f"{image_path.stem}_processed{image_path.suffix}"
                cv2.imwrite(str(output_path), results['final'])
                
                # Collect statistics
                statistics['successful'] += 1
                all_metrics.append(results['metrics'])
                
                # Track pipeline usage
                pipeline_key = ' -> '.join(results['pipeline'])
                statistics['pipeline_usage'][pipeline_key] = \
                    statistics['pipeline_usage'].get(pipeline_key, 0) + 1
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                statistics['failed'].append({
                    'file': image_path.name,
                    'error': str(e)
                })
        
        # Calculate average metrics
        if all_metrics:
            avg_metrics = {
                'contrast': float(np.mean([m.contrast for m in all_metrics])),
                'brightness': float(np.mean([m.brightness for m in all_metrics])),
                'sharpness': float(np.mean([m.sharpness for m in all_metrics])),
                'noise_level': float(np.mean([m.noise_level for m in all_metrics])),
                'avg_skew': float(np.mean([abs(m.skew_angle) for m in all_metrics]))
            }
            statistics['average_metrics'] = avg_metrics
        
        return statistics
    
    def optimize_for_ocr_engine(self, image_path: Path, ocr_engine: str = 'tesseract') -> np.ndarray:
        """Optimize preprocessing specifically for chosen OCR engine."""
        engine_configs = {
            'tesseract': {
                'target_dpi': 300,
                'prefer_binary': True,
                'contrast_boost': 1.2
            },
            'google_vision': {
                'target_dpi': 200,
                'prefer_binary': False,
                'contrast_boost': 1.0
            }
        }
        
        config = engine_configs.get(ocr_engine, engine_configs['tesseract'])
        
        # Load and process image
        image = cv2.imread(str(image_path))
        
        # Apply engine-specific optimizations
        if config['contrast_boost'] != 1.0:
            image = cv2.convertScaleAbs(image, alpha=config['contrast_boost'], beta=0)
        
        # Process with standard pipeline
        results = self.process_image(image_path)
        
        # Apply final engine-specific adjustments
        final_image = results['final']
        
        if config['prefer_binary'] and len(final_image.shape) == 3:
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            _, final_image = cv2.threshold(final_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return final_image


def create_comparison_report(preprocessor: DocumentPreprocessor, 
                          test_folder: Path,
                          output_path: Path) -> None:
    """Create visual comparison report of preprocessing effects."""
    
    # Select sample images
    sample_images = list(test_folder.glob("*.JPG"))[:5]
    
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 4*len(sample_images)))
    fig.suptitle('Preprocessing Pipeline Comparison', fontsize=16)
    
    for idx, image_path in enumerate(sample_images):
        # Process image
        results = preprocessor.process_image(image_path, save_intermediate=True)
        
        # Original
        axes[idx, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'Original\n{image_path.name}')
        axes[idx, 0].axis('off')
        
        # After contrast enhancement
        if 'enhance_contrast' in results['steps']:
            axes[idx, 1].imshow(cv2.cvtColor(results['steps']['enhance_contrast'], cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title('Enhanced Contrast')
        else:
            axes[idx, 1].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title('No Enhancement')
        axes[idx, 1].axis('off')
        
        # After deskew
        if 'deskew' in results['steps']:
            axes[idx, 2].imshow(cv2.cvtColor(results['steps']['deskew'], cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title(f'Deskewed ({results["metrics"].skew_angle:.1f}°)')
        else:
            axes[idx, 2].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title('No Deskew Needed')
        axes[idx, 2].axis('off')
        
        # Final binary
        final = results['final']
        if len(final.shape) == 2:
            axes[idx, 3].imshow(final, cmap='gray')
        else:
            axes[idx, 3].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        axes[idx, 3].set_title('Final (Binary)')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison report saved to {output_path}")


def main():
    """Test preprocessing pipeline on Hansman Syracuse collection."""
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor()
    
    # Set up paths
    input_folder = Path("/workspaces/amy-project/hansman_organized/downloads")
    output_folder = Path("/workspaces/amy-project/preprocessing_results")
    
    # Process batch
    print("Testing preprocessing pipeline on Hansman Syracuse photos...")
    stats = preprocessor.batch_process(input_folder, output_folder, sample_size=10)
    
    # Save statistics
    stats_path = output_folder / "preprocessing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Success rate: {stats['successful']}/{stats['total_files']}")
    print(f"Results saved to: {output_folder}")
    
    # Create visual comparison
    comparison_path = output_folder / "preprocessing_comparison.png"
    create_comparison_report(preprocessor, input_folder, comparison_path)
    
    # Print pipeline usage statistics
    print("\nPipeline Usage:")
    for pipeline, count in stats['pipeline_usage'].items():
        print(f"  {pipeline}: {count} times")
    
    # Print average metrics
    if 'average_metrics' in stats:
        print("\nAverage Image Metrics:")
        for metric, value in stats['average_metrics'].items():
            print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    main()