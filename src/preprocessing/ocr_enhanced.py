"""
Enhanced Tesseract OCR integration for hybrid document extraction system.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pytesseract
import cv2
from PIL import Image

from ..file_access.ocr_processor import OCRProcessor
from .enhancer import ImageEnhancer
from .metadata import ProcessingStep

logger = logging.getLogger(__name__)


@dataclass
class CharacterConfidence:
    """Character-level confidence information."""

    character: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    page_number: int = 1
    line_number: int = 0
    word_number: int = 0


@dataclass
class WordConfidence:
    """Word-level confidence information."""

    word: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    page_number: int = 1
    line_number: int = 0
    characters: List[CharacterConfidence] = field(default_factory=list)


@dataclass
class LineData:
    """Line-level data with position and confidence."""

    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    words: List[WordConfidence]
    baseline: int
    skew_angle: float = 0.0


@dataclass
class ProblematicRegion:
    """Region with low confidence or issues."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    issue_type: str  # 'low_confidence', 'unusual_characters', 'skewed', 'blurred'
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""


@dataclass
class EnhancedOCRResult:
    """Enhanced OCR result with detailed confidence and position data."""

    text: str
    overall_confidence: float
    word_confidences: List[WordConfidence]
    character_confidences: List[CharacterConfidence]
    lines: List[LineData]
    problematic_regions: List[ProblematicRegion]
    processing_time: float
    variant_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "overall_confidence": self.overall_confidence,
            "word_confidences": [asdict(w) for w in self.word_confidences],
            "character_confidences": [asdict(c) for c in self.character_confidences],
            "lines": [
                {
                    "text": line.text,
                    "bbox": line.bbox,
                    "confidence": line.confidence,
                    "words": [asdict(w) for w in line.words],
                    "baseline": line.baseline,
                    "skew_angle": line.skew_angle,
                }
                for line in self.lines
            ],
            "problematic_regions": [asdict(r) for r in self.problematic_regions],
            "processing_time": self.processing_time,
            "variant_results": self.variant_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedOCRResult":
        """Create from dictionary."""
        word_confidences = [
            WordConfidence(**w) for w in data.get("word_confidences", [])
        ]
        character_confidences = [
            CharacterConfidence(**c) for c in data.get("character_confidences", [])
        ]

        lines = []
        for line_data in data.get("lines", []):
            line_words = [WordConfidence(**w) for w in line_data.get("words", [])]
            lines.append(
                LineData(
                    text=line_data["text"],
                    bbox=tuple(line_data["bbox"]),
                    confidence=line_data["confidence"],
                    words=line_words,
                    baseline=line_data["baseline"],
                    skew_angle=line_data.get("skew_angle", 0.0),
                )
            )

        problematic_regions = [
            ProblematicRegion(**r) for r in data.get("problematic_regions", [])
        ]

        return cls(
            text=data["text"],
            overall_confidence=data["overall_confidence"],
            word_confidences=word_confidences,
            character_confidences=character_confidences,
            lines=lines,
            problematic_regions=problematic_regions,
            processing_time=data["processing_time"],
            variant_results=data.get("variant_results", {}),
            metadata=data.get("metadata", {}),
        )


class EnhancedOCRProcessor:
    """Enhanced OCR processor for hybrid document extraction."""

    def __init__(
        self,
        language: str = "eng",
        confidence_threshold: float = 60.0,
        parallel_processing: bool = True,
        max_workers: int = 4,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the enhanced OCR processor.

        Args:
            language: Tesseract language code
            confidence_threshold: Minimum confidence threshold
            parallel_processing: Enable parallel processing
            max_workers: Maximum worker threads
            cache_enabled: Enable result caching
            cache_dir: Directory for cache storage
        """
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled

        if cache_enabled and cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Initialize base OCR processor
        self.base_processor = OCRProcessor(language=language)

        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")

    def process_variants(
        self, image_variants: Dict[str, np.ndarray], original_path: str
    ) -> EnhancedOCRResult:
        """Process multiple image variants and combine results.

        Args:
            image_variants: Dictionary of image variant names to arrays
            original_path: Path to original image

        Returns:
            Enhanced OCR result combining all variants
        """
        start_time = time.time()

        # Check cache first
        if self.cache_enabled and self.cache_dir:
            cache_key = self._get_cache_key(original_path)
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                return cached_result

        # Process each variant
        variant_results = {}

        if self.parallel_processing and len(image_variants) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_variant = {
                    executor.submit(
                        self._process_single_variant, variant_name, image_array
                    ): variant_name
                    for variant_name, image_array in image_variants.items()
                }

                for future in as_completed(future_to_variant):
                    variant_name = future_to_variant[future]
                    try:
                        result = future.result()
                        variant_results[variant_name] = result
                    except Exception as e:
                        logger.error(f"Error processing variant {variant_name}: {e}")
        else:
            for variant_name, image_array in image_variants.items():
                try:
                    result = self._process_single_variant(variant_name, image_array)
                    variant_results[variant_name] = result
                except Exception as e:
                    logger.error(f"Error processing variant {variant_name}: {e}")

        # Combine results from all variants
        combined_result = self._combine_variant_results(variant_results)

        # Identify problematic regions
        problematic_regions = self._identify_problematic_regions(combined_result)
        combined_result.problematic_regions = problematic_regions

        # Add processing time
        combined_result.processing_time = time.time() - start_time

        # Save to cache
        if self.cache_enabled and self.cache_dir:
            self._save_to_cache(cache_key, combined_result)

        return combined_result

    def _process_single_variant(
        self, variant_name: str, image_array: np.ndarray
    ) -> Dict[str, Any]:
        """Process a single image variant with Tesseract.

        Args:
            variant_name: Name of the variant
            image_array: Image array to process

        Returns:
            OCR results for this variant
        """
        # Convert numpy array to PIL Image for Tesseract
        if len(image_array.shape) == 3:
            image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image_array)

        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            image, lang=self.language, output_type=pytesseract.Output.DICT
        )

        # Get HOCR output for additional details
        hocr_data = pytesseract.image_to_pdf_or_hocr(
            image, lang=self.language, extension="hocr"
        )

        # Parse and structure the results
        word_confidences = []
        character_confidences = []
        lines = []

        current_line = None
        current_line_words = []

        for i in range(len(ocr_data["text"])):
            if ocr_data["text"][i].strip():
                # Word-level data
                word = ocr_data["text"][i]
                confidence = float(ocr_data["conf"][i])

                bbox = (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                )

                word_conf = WordConfidence(
                    word=word,
                    confidence=confidence,
                    bbox=bbox,
                    page_number=ocr_data["page_num"][i],
                    line_number=ocr_data["line_num"][i],
                )

                word_confidences.append(word_conf)

                # Group words into lines
                if current_line is None or ocr_data["line_num"][i] != current_line:
                    if current_line is not None and current_line_words:
                        # Complete the previous line
                        line_text = " ".join([w.word for w in current_line_words])
                        line_bbox = self._merge_bboxes(
                            [w.bbox for w in current_line_words]
                        )
                        line_confidence = np.mean(
                            [w.confidence for w in current_line_words]
                        )

                        lines.append(
                            LineData(
                                text=line_text,
                                bbox=line_bbox,
                                confidence=line_confidence,
                                words=current_line_words,
                                baseline=line_bbox[1] + line_bbox[3],
                            )
                        )

                    current_line = ocr_data["line_num"][i]
                    current_line_words = []

                current_line_words.append(word_conf)

        # Handle the last line
        if current_line_words:
            line_text = " ".join([w.word for w in current_line_words])
            line_bbox = self._merge_bboxes([w.bbox for w in current_line_words])
            line_confidence = np.mean([w.confidence for w in current_line_words])

            lines.append(
                LineData(
                    text=line_text,
                    bbox=line_bbox,
                    confidence=line_confidence,
                    words=current_line_words,
                    baseline=line_bbox[1] + line_bbox[3],
                )
            )

        # Get character-level data if available
        try:
            char_data = pytesseract.image_to_boxes(
                image, lang=self.language, output_type=pytesseract.Output.DICT
            )

            for i in range(len(char_data["char"])):
                char_conf = CharacterConfidence(
                    character=char_data["char"][i],
                    confidence=100.0,  # Tesseract doesn't provide char confidence
                    bbox=(
                        char_data["left"][i],
                        char_data["bottom"][i],
                        char_data["right"][i] - char_data["left"][i],
                        char_data["top"][i] - char_data["bottom"][i],
                    ),
                )
                character_confidences.append(char_conf)
        except Exception as e:
            logger.warning(f"Could not extract character-level data: {e}")

        return {
            "variant_name": variant_name,
            "text": pytesseract.image_to_string(image, lang=self.language),
            "word_confidences": word_confidences,
            "character_confidences": character_confidences,
            "lines": lines,
            "raw_data": ocr_data,
        }

    def _combine_variant_results(
        self, variant_results: Dict[str, Dict[str, Any]]
    ) -> EnhancedOCRResult:
        """Combine results from multiple variants.

        Args:
            variant_results: Results from each variant

        Returns:
            Combined enhanced OCR result
        """
        # Select best text based on confidence
        best_text = ""
        best_confidence = 0.0
        all_word_confidences = []
        all_lines = []

        for variant_name, result in variant_results.items():
            # Calculate average confidence for this variant
            if result["word_confidences"]:
                avg_confidence = np.mean(
                    [w.confidence for w in result["word_confidences"]]
                )

                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = result["text"]
                    all_word_confidences = result["word_confidences"]
                    all_lines = result["lines"]

        # Combine character confidences from all variants
        all_char_confidences = []
        for result in variant_results.values():
            all_char_confidences.extend(result.get("character_confidences", []))

        # Remove duplicates
        unique_chars = {}
        for char in all_char_confidences:
            key = (char.character, char.bbox)
            if (
                key not in unique_chars
                or char.confidence > unique_chars[key].confidence
            ):
                unique_chars[key] = char

        return EnhancedOCRResult(
            text=best_text,
            overall_confidence=best_confidence,
            word_confidences=all_word_confidences,
            character_confidences=list(unique_chars.values()),
            lines=all_lines,
            problematic_regions=[],
            processing_time=0.0,
            variant_results={k: v["text"] for k, v in variant_results.items()},
        )

    def _identify_problematic_regions(
        self, result: EnhancedOCRResult
    ) -> List[ProblematicRegion]:
        """Identify regions with potential issues.

        Args:
            result: Enhanced OCR result

        Returns:
            List of problematic regions
        """
        problematic_regions = []

        # Identify low confidence regions
        for word in result.word_confidences:
            if word.confidence < self.confidence_threshold:
                region = ProblematicRegion(
                    bbox=word.bbox,
                    confidence=word.confidence,
                    issue_type="low_confidence",
                    details={"word": word.word},
                    suggested_action="Review with Claude AI",
                )
                problematic_regions.append(region)

        # Identify regions with unusual characters
        import re

        unusual_pattern = re.compile(r"[^\w\s\.\,\!\?\-\:\;\'\"\(\)]")

        for word in result.word_confidences:
            if unusual_pattern.search(word.word):
                region = ProblematicRegion(
                    bbox=word.bbox,
                    confidence=word.confidence,
                    issue_type="unusual_characters",
                    details={
                        "word": word.word,
                        "characters": unusual_pattern.findall(word.word),
                    },
                    suggested_action="Verify with image analysis",
                )
                problematic_regions.append(region)

        # Identify potentially skewed text
        if result.lines:
            baselines = [line.baseline for line in result.lines]
            if len(baselines) > 1:
                # Check if baselines are not aligned
                baseline_variance = np.var(baselines)
                if baseline_variance > 100:  # Threshold for skew detection
                    region = ProblematicRegion(
                        bbox=self._merge_bboxes([line.bbox for line in result.lines]),
                        confidence=result.overall_confidence,
                        issue_type="skewed",
                        details={"variance": baseline_variance},
                        suggested_action="Apply deskew correction",
                    )
                    problematic_regions.append(region)

        return problematic_regions

    def _merge_bboxes(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Merge multiple bounding boxes into one.

        Args:
            bboxes: List of bounding boxes (x, y, width, height)

        Returns:
            Merged bounding box
        """
        if not bboxes:
            return (0, 0, 0, 0)

        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for an image.

        Args:
            image_path: Path to the image

        Returns:
            Cache key
        """
        import hashlib

        content = f"{image_path}:{self.language}:{self.confidence_threshold}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[EnhancedOCRResult]:
        """Load results from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached result if available
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                return EnhancedOCRResult.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: EnhancedOCRResult):
        """Save results to cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, "item"):  # numpy scalar
                        return obj.item()
                    if hasattr(obj, "tolist"):  # numpy array
                        return obj.tolist()
                    return super().default(obj)

            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def export_structured_output(
        self, result: EnhancedOCRResult, output_path: str, format: str = "json"
    ):
        """Export OCR results in structured format.

        Args:
            result: Enhanced OCR result
            output_path: Path to save output
            format: Output format ('json', 'xml', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, "item"):  # numpy scalar
                        return obj.item()
                    if hasattr(obj, "tolist"):  # numpy array
                        return obj.tolist()
                    return super().default(obj)

            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)

        elif format == "xml":
            import xml.etree.ElementTree as ET

            root = ET.Element("OCRResult")
            root.set("confidence", str(result.overall_confidence))

            # Add text
            text_elem = ET.SubElement(root, "Text")
            text_elem.text = result.text

            # Add words
            words_elem = ET.SubElement(root, "Words")
            for word in result.word_confidences:
                word_elem = ET.SubElement(words_elem, "Word")
                word_elem.set("confidence", str(word.confidence))
                word_elem.set("bbox", ",".join(map(str, word.bbox)))
                word_elem.text = word.word

            # Add lines
            lines_elem = ET.SubElement(root, "Lines")
            for line in result.lines:
                line_elem = ET.SubElement(lines_elem, "Line")
                line_elem.set("confidence", str(line.confidence))
                line_elem.set("bbox", ",".join(map(str, line.bbox)))
                line_elem.text = line.text

            tree = ET.ElementTree(root)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Type", "Text", "Confidence", "X", "Y", "Width", "Height"]
                )

                for word in result.word_confidences:
                    writer.writerow(
                        [
                            "word",
                            word.word,
                            word.confidence,
                            word.bbox[0],
                            word.bbox[1],
                            word.bbox[2],
                            word.bbox[3],
                        ]
                    )

                for line in result.lines:
                    writer.writerow(
                        [
                            "line",
                            line.text,
                            line.confidence,
                            line.bbox[0],
                            line.bbox[1],
                            line.bbox[2],
                            line.bbox[3],
                        ]
                    )

        logger.info(f"Exported OCR results to {output_path} in {format} format")
