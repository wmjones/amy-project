"""
OCR processing utilities for document images.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import cv2

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Advanced OCR processing for document images."""

    def __init__(
        self,
        language: str = "eng",
        dpi: int = 300,
        enhance_contrast: bool = True,
        denoise: bool = True,
        deskew: bool = True,
    ):
        """Initialize OCR processor.

        Args:
            language: OCR language code (e.g., 'eng', 'fra', 'deu')
            dpi: Target DPI for OCR
            enhance_contrast: Whether to enhance image contrast
            denoise: Whether to denoise image
            deskew: Whether to deskew image
        """
        self.language = language
        self.dpi = dpi
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.deskew = deskew

        # Verify Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image for OCR.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with OCR results and metadata
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Apply preprocessing
                processed_img = self._preprocess_image(img)

                # Perform OCR
                ocr_data = self._perform_ocr(processed_img)

                # Get additional metadata
                metadata = self._extract_metadata(processed_img, ocr_data)

                return {
                    "success": True,
                    "text": ocr_data["text"],
                    "confidence": ocr_data["confidence"],
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {"success": False, "error": str(e), "text": "", "confidence": 0}

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if not already
        if img.mode != "L":
            img = img.convert("L")

        # Resize for target DPI if needed
        img = self._resize_for_dpi(img)

        # Apply image enhancements
        if self.enhance_contrast:
            img = self._enhance_contrast(img)

        # Convert to numpy array for OpenCV operations
        img_array = np.array(img)

        if self.denoise:
            img_array = self._denoise_image(img_array)

        if self.deskew:
            img_array = self._deskew_image(img_array)

        # Convert back to PIL Image
        return Image.fromarray(img_array)

    def _resize_for_dpi(self, img: Image.Image, target_dpi: int = 300) -> Image.Image:
        """Resize image to target DPI."""
        # Get current DPI (default to 72 if not specified)
        current_dpi = img.info.get("dpi", (72, 72))[0]

        if current_dpi < target_dpi:
            scale_factor = target_dpi / current_dpi
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _enhance_contrast(self, img: Image.Image) -> Image.Image:
        """Enhance image contrast."""
        # Apply adaptive histogram equalization
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Apply sharpening
        img = img.filter(ImageFilter.SHARPEN)

        return img

    def _denoise_image(self, img_array: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(img_array)

        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        return denoised

    def _deskew_image(self, img_array: np.ndarray) -> np.ndarray:
        """Deskew tilted image."""
        # Detect text orientation
        coords = np.column_stack(np.where(img_array > 0))

        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Rotate image
            (h, w) = img_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img_array,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            return rotated

        return img_array

    def _perform_ocr(self, img: Image.Image) -> Dict[str, Any]:
        """Perform OCR on preprocessed image."""
        try:
            # Get OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                img, lang=self.language, output_type=pytesseract.Output.DICT
            )

            # Extract text and calculate average confidence
            text = pytesseract.image_to_string(img, lang=self.language)

            # Calculate average confidence (excluding -1 values)
            confidences = [c for c in ocr_data["conf"] if c != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": text,
                "confidence": avg_confidence,
                "word_count": len([w for w in ocr_data["text"] if w.strip()]),
                "data": ocr_data,
            }

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {"text": "", "confidence": 0, "word_count": 0, "error": str(e)}

    def _extract_metadata(self, img: Image.Image, ocr_data: Dict) -> Dict[str, Any]:
        """Extract additional metadata from OCR results."""
        metadata = {
            "image_size": img.size,
            "mode": img.mode,
            "format": img.format,
            "word_count": ocr_data.get("word_count", 0),
            "avg_confidence": ocr_data.get("confidence", 0),
        }

        # Detect document type based on content
        text_lower = ocr_data.get("text", "").lower()

        if any(term in text_lower for term in ["invoice", "bill", "receipt"]):
            metadata["document_type"] = "financial"
        elif any(term in text_lower for term in ["contract", "agreement", "terms"]):
            metadata["document_type"] = "legal"
        elif any(term in text_lower for term in ["report", "analysis", "summary"]):
            metadata["document_type"] = "report"
        else:
            metadata["document_type"] = "general"

        return metadata

    def batch_process(self, image_paths: list, progress_callback=None) -> list:
        """Process multiple images in batch.

        Args:
            image_paths: List of image file paths
            progress_callback: Optional progress callback function

        Returns:
            List of OCR results
        """
        results = []
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            result = self.process_image(image_path)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def detect_text_regions(self, image_path: str) -> list:
        """Detect text regions in an image.

        Args:
            image_path: Path to image file

        Returns:
            List of text regions with bounding boxes
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != "L":
                    img = img.convert("L")

                # Get bounding boxes
                boxes = pytesseract.image_to_boxes(img, lang=self.language)

                # Parse boxes
                regions = []
                for box in boxes.splitlines():
                    parts = box.split()
                    if len(parts) >= 5:
                        char = parts[0]
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        conf = int(parts[5]) if len(parts) > 5 else 0

                        regions.append(
                            {"char": char, "box": (x1, y1, x2, y2), "confidence": conf}
                        )

                return regions

        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
