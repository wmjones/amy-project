"""
Image enhancement algorithms for document preprocessing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancementParameters:
    """Parameters for image enhancement operations."""

    deblur_strength: float = 1.0
    contrast_factor: float = 1.5
    brightness_factor: float = 1.0
    binarization_threshold: int = 127
    adaptive_binarization: bool = True
    denoising_strength: int = 3
    sharpening_amount: float = 1.0


class ImageEnhancer:
    """Image enhancement processor for document images."""

    def __init__(self, params: Optional[EnhancementParameters] = None):
        """Initialize the image enhancer.

        Args:
            params: Enhancement parameters configuration
        """
        self.params = params or EnhancementParameters()

    def enhance_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Apply various enhancement algorithms to a document image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary of enhanced image variants with keys:
                - 'original': Original image
                - 'deblurred': Deblurred version
                - 'contrast_enhanced': Contrast enhanced version
                - 'binarized': Binary (black & white) version
                - 'denoised': Noise-reduced version
                - 'sharpened': Sharpened version
                - 'combined': Best combined enhancement
        """
        logger.info(f"Enhancing image: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = {"original": image}

        # Apply enhancement algorithms
        try:
            results["deblurred"] = self._deblur_image(gray)
            results["contrast_enhanced"] = self._enhance_contrast(gray)
            results["binarized"] = self._binarize_image(gray)
            results["denoised"] = self._denoise_image(gray)
            results["sharpened"] = self._sharpen_image(gray)
            results["combined"] = self._combine_enhancements(gray)
        except Exception as e:
            logger.error(f"Error during enhancement: {e}")
            raise

        return results

    def _deblur_image(self, image: np.ndarray) -> np.ndarray:
        """Apply deblurring to reduce motion blur or focus blur.

        Args:
            image: Grayscale image array

        Returns:
            Deblurred image
        """
        logger.debug("Applying deblurring")

        # Use Wiener deconvolution approximation
        kernel_size = int(5 * self.params.deblur_strength)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create a simple sharpening kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2] = 2.0
        gaussian = cv2.getGaussianKernel(kernel_size, 1.0)
        gaussian_2d = gaussian @ gaussian.T
        kernel -= gaussian_2d

        # Apply the kernel
        deblurred = cv2.filter2D(image, -1, kernel)

        # Limit the enhancement to avoid artifacts
        deblurred = cv2.addWeighted(image, 0.5, deblurred, 0.5, 0)

        return deblurred

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better text visibility.

        Args:
            image: Grayscale image array

        Returns:
            Contrast enhanced image
        """
        logger.debug("Enhancing contrast")

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=2.0 * self.params.contrast_factor, tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(image)

        # Additional global contrast adjustment
        pil_image = Image.fromarray(enhanced)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_enhanced = enhancer.enhance(self.params.contrast_factor)

        return np.array(pil_enhanced)

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white) for clear text.

        Args:
            image: Grayscale image array

        Returns:
            Binary image
        """
        logger.debug("Applying binarization")

        if self.params.adaptive_binarization:
            # Adaptive thresholding for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Simple Otsu's thresholding
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from the image for cleaner text.

        Args:
            image: Grayscale image array

        Returns:
            Denoised image
        """
        logger.debug("Removing noise")

        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self.params.denoising_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        # Additional median filter for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)

        return denoised

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening to enhance text edges.

        Args:
            image: Grayscale image array

        Returns:
            Sharpened image
        """
        logger.debug("Sharpening image")

        # Create sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

        # Scale kernel by sharpening amount
        kernel = kernel * self.params.sharpening_amount
        kernel[1, 1] = 4 + self.params.sharpening_amount

        sharpened = cv2.filter2D(image, -1, kernel)

        # Limit the enhancement
        sharpened = cv2.addWeighted(
            image,
            1 - self.params.sharpening_amount * 0.5,
            sharpened,
            self.params.sharpening_amount * 0.5,
            0,
        )

        return sharpened

    def _combine_enhancements(self, image: np.ndarray) -> np.ndarray:
        """Combine multiple enhancement techniques for optimal results.

        Args:
            image: Grayscale image array

        Returns:
            Combined enhanced image
        """
        logger.debug("Combining enhancements")

        # Start with denoising
        result = self._denoise_image(image)

        # Enhance contrast
        result = self._enhance_contrast(result)

        # Apply mild sharpening
        result = self._sharpen_image(result)

        # Optional: Apply very mild deblurring
        if self.params.deblur_strength > 0:
            deblurred = self._deblur_image(result)
            result = cv2.addWeighted(result, 0.7, deblurred, 0.3, 0)

        return result

    def batch_enhance(self, image_paths: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Process multiple images in batch.

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary mapping image paths to their enhancement results
        """
        results = {}
        for path in image_paths:
            try:
                results[path] = self.enhance_image(path)
            except Exception as e:
                logger.error(f"Failed to enhance {path}: {e}")
                results[path] = {"error": str(e)}

        return results

    def save_enhanced_images(
        self,
        enhanced_images: Dict[str, np.ndarray],
        output_dir: str,
        base_filename: str,
    ) -> Dict[str, str]:
        """Save enhanced images to disk.

        Args:
            enhanced_images: Dictionary of enhancement type to image array
            output_dir: Directory to save images
            base_filename: Base name for output files

        Returns:
            Dictionary mapping enhancement type to saved file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for enhancement_type, image in enhanced_images.items():
            if enhancement_type == "error":
                continue

            filename = f"{base_filename}_{enhancement_type}.png"
            output_path = output_dir / filename

            if len(image.shape) == 2:  # Grayscale
                cv2.imwrite(str(output_path), image)
            else:  # Color
                cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            saved_paths[enhancement_type] = str(output_path)
            logger.debug(f"Saved {enhancement_type} to {output_path}")

        return saved_paths
