"""
Unit tests for enhanced OCR processor.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from src.preprocessing.ocr_enhanced import (
    EnhancedOCRProcessor,
    EnhancedOCRResult,
    CharacterConfidence,
    WordConfidence,
    LineData,
    ProblematicRegion,
)
from PIL import Image


@pytest.fixture
def mock_tesseract_data():
    """Mock Tesseract OCR data."""
    return {
        "text": ["Hello", "World", "", "Test", "Document"],
        "conf": [95, 90, -1, 85, 80],
        "left": [10, 60, 0, 10, 60],
        "top": [10, 10, 0, 50, 50],
        "width": [40, 40, 0, 30, 50],
        "height": [20, 20, 0, 20, 20],
        "page_num": [1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 2, 2],
    }


@pytest.fixture
def mock_char_data():
    """Mock character-level data."""
    return {
        "char": ["H", "e", "l", "l", "o"],
        "left": [10, 14, 18, 22, 26],
        "bottom": [10, 10, 10, 10, 10],
        "right": [14, 18, 22, 26, 30],
        "top": [30, 30, 30, 30, 30],
    }


@pytest.fixture
def test_image_array():
    """Create a test image array."""
    # Create a simple black text on white background image
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Test Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


@pytest.fixture
def enhanced_ocr_processor():
    """Create an enhanced OCR processor instance."""
    return EnhancedOCRProcessor(
        language="eng",
        confidence_threshold=70.0,
        parallel_processing=False,
        cache_enabled=False,
    )


class TestEnhancedOCRProcessor:
    """Test the EnhancedOCRProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = EnhancedOCRProcessor()
        assert processor.language == "eng"
        assert processor.confidence_threshold == 60.0
        assert processor.parallel_processing is True

    def test_initialization_with_cache(self, tmp_path):
        """Test processor initialization with cache enabled."""
        cache_dir = tmp_path / "ocr_cache"
        processor = EnhancedOCRProcessor(cache_enabled=True, cache_dir=str(cache_dir))
        assert processor.cache_enabled is True
        assert processor.cache_dir == cache_dir
        assert cache_dir.exists()

    @patch("pytesseract.image_to_data")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_pdf_or_hocr")
    @patch("pytesseract.image_to_boxes")
    def test_process_single_variant(
        self,
        mock_boxes,
        mock_hocr,
        mock_string,
        mock_data,
        enhanced_ocr_processor,
        test_image_array,
        mock_tesseract_data,
        mock_char_data,
    ):
        """Test processing a single image variant."""
        # Setup mocks
        mock_data.return_value = mock_tesseract_data
        mock_string.return_value = "Hello World Test Document"
        mock_hocr.return_value = b"<html>mock hocr</html>"
        mock_boxes.return_value = mock_char_data

        # Process variant
        result = enhanced_ocr_processor._process_single_variant(
            "test_variant", test_image_array
        )

        assert result["variant_name"] == "test_variant"
        assert result["text"] == "Hello World Test Document"
        assert len(result["word_confidences"]) == 4  # Excluding empty text
        assert result["word_confidences"][0].word == "Hello"
        assert result["word_confidences"][0].confidence == 95
        assert len(result["lines"]) == 2

    def test_combine_variant_results(self, enhanced_ocr_processor):
        """Test combining results from multiple variants."""
        # Create mock variant results
        variant_results = {
            "variant1": {
                "text": "Text from variant 1",
                "word_confidences": [
                    WordConfidence("Text", 90, (0, 0, 10, 10)),
                    WordConfidence("from", 85, (15, 0, 10, 10)),
                ],
                "character_confidences": [],
                "lines": [
                    LineData(
                        text="Text from",
                        bbox=(0, 0, 25, 10),
                        confidence=87.5,
                        words=[],
                        baseline=10,
                    )
                ],
            },
            "variant2": {
                "text": "Text from variant 2",
                "word_confidences": [
                    WordConfidence("Text", 95, (0, 0, 10, 10)),
                    WordConfidence("from", 92, (15, 0, 10, 10)),
                ],
                "character_confidences": [],
                "lines": [
                    LineData(
                        text="Text from",
                        bbox=(0, 0, 25, 10),
                        confidence=93.5,
                        words=[],
                        baseline=10,
                    )
                ],
            },
        }

        result = enhanced_ocr_processor._combine_variant_results(variant_results)

        # Should choose variant2 as it has higher confidence
        assert result.text == "Text from variant 2"
        assert result.overall_confidence == 93.5
        assert len(result.word_confidences) == 2
        assert result.word_confidences[0].confidence == 95

    def test_identify_problematic_regions(self, enhanced_ocr_processor):
        """Test identification of problematic regions."""
        # Create a result with various issues
        result = EnhancedOCRResult(
            text="Hello W@rld Test",
            overall_confidence=80,
            word_confidences=[
                WordConfidence("Hello", 95, (0, 0, 30, 20)),
                WordConfidence("W@rld", 50, (40, 0, 30, 20)),  # Low confidence
                WordConfidence("Test", 85, (80, 0, 30, 20)),
            ],
            character_confidences=[],
            lines=[
                LineData(
                    text="Hello W@rld Test",
                    bbox=(0, 0, 110, 20),
                    confidence=76.7,
                    words=[],
                    baseline=20,
                )
            ],
            problematic_regions=[],
            processing_time=0.0,
        )

        regions = enhanced_ocr_processor._identify_problematic_regions(result)

        assert len(regions) >= 2  # At least low confidence and unusual character

        # Check for low confidence region
        low_conf_regions = [r for r in regions if r.issue_type == "low_confidence"]
        assert len(low_conf_regions) == 1
        assert low_conf_regions[0].details["word"] == "W@rld"

        # Check for unusual character region
        unusual_char_regions = [
            r for r in regions if r.issue_type == "unusual_characters"
        ]
        assert len(unusual_char_regions) == 1
        assert "@" in unusual_char_regions[0].details["characters"]

    def test_merge_bboxes(self, enhanced_ocr_processor):
        """Test bounding box merging."""
        bboxes = [(10, 20, 30, 40), (40, 25, 30, 35), (70, 30, 20, 30)]

        merged = enhanced_ocr_processor._merge_bboxes(bboxes)

        assert merged[0] == 10  # min x
        assert merged[1] == 20  # min y
        assert merged[2] == 80  # width (90 - 10)
        assert merged[3] == 40  # height (60 - 20)

    def test_cache_functionality(self, enhanced_ocr_processor, tmp_path):
        """Test caching functionality."""
        cache_dir = tmp_path / "cache"
        enhanced_ocr_processor.cache_enabled = True
        enhanced_ocr_processor.cache_dir = cache_dir
        cache_dir.mkdir()

        # Create a test result
        test_result = EnhancedOCRResult(
            text="Test text",
            overall_confidence=90.0,
            word_confidences=[],
            character_confidences=[],
            lines=[],
            problematic_regions=[],
            processing_time=1.0,
        )

        # Save to cache
        cache_key = enhanced_ocr_processor._get_cache_key("test_image.png")
        enhanced_ocr_processor._save_to_cache(cache_key, test_result)

        # Load from cache
        loaded_result = enhanced_ocr_processor._load_from_cache(cache_key)

        assert loaded_result is not None
        assert loaded_result.text == test_result.text
        assert loaded_result.overall_confidence == test_result.overall_confidence

    def test_export_structured_output(self, enhanced_ocr_processor, tmp_path):
        """Test exporting structured output."""
        # Create a test result
        test_result = EnhancedOCRResult(
            text="Test text",
            overall_confidence=90.0,
            word_confidences=[
                WordConfidence("Test", 92, (0, 0, 30, 20)),
                WordConfidence("text", 88, (35, 0, 30, 20)),
            ],
            character_confidences=[],
            lines=[
                LineData(
                    text="Test text",
                    bbox=(0, 0, 65, 20),
                    confidence=90,
                    words=[],
                    baseline=20,
                )
            ],
            problematic_regions=[],
            processing_time=1.0,
        )

        # Test JSON export
        json_path = tmp_path / "output.json"
        enhanced_ocr_processor.export_structured_output(
            test_result, str(json_path), format="json"
        )
        assert json_path.exists()

        with open(json_path, "r") as f:
            data = json.load(f)
            assert data["text"] == "Test text"
            assert data["overall_confidence"] == 90.0

        # Test XML export
        xml_path = tmp_path / "output.xml"
        enhanced_ocr_processor.export_structured_output(
            test_result, str(xml_path), format="xml"
        )
        assert xml_path.exists()

        # Test CSV export
        csv_path = tmp_path / "output.csv"
        enhanced_ocr_processor.export_structured_output(
            test_result, str(csv_path), format="csv"
        )
        assert csv_path.exists()


class TestDataClasses:
    """Test the data classes."""

    def test_character_confidence(self):
        """Test CharacterConfidence dataclass."""
        char = CharacterConfidence(character="A", confidence=95.0, bbox=(10, 20, 5, 10))
        assert char.character == "A"
        assert char.confidence == 95.0
        assert char.bbox == (10, 20, 5, 10)

    def test_word_confidence(self):
        """Test WordConfidence dataclass."""
        word = WordConfidence(
            word="Hello",
            confidence=90.0,
            bbox=(0, 0, 50, 20),
            page_number=1,
            line_number=1,
        )
        assert word.word == "Hello"
        assert word.confidence == 90.0
        assert word.page_number == 1

    def test_line_data(self):
        """Test LineData dataclass."""
        line = LineData(
            text="Test line",
            bbox=(0, 0, 100, 20),
            confidence=88.5,
            words=[],
            baseline=20,
            skew_angle=0.5,
        )
        assert line.text == "Test line"
        assert line.confidence == 88.5
        assert line.skew_angle == 0.5

    def test_problematic_region(self):
        """Test ProblematicRegion dataclass."""
        region = ProblematicRegion(
            bbox=(10, 20, 30, 40),
            confidence=45.0,
            issue_type="low_confidence",
            details={"word": "unclear"},
            suggested_action="Review manually",
        )
        assert region.issue_type == "low_confidence"
        assert region.suggested_action == "Review manually"

    def test_enhanced_ocr_result_serialization(self):
        """Test EnhancedOCRResult serialization."""
        result = EnhancedOCRResult(
            text="Test",
            overall_confidence=90.0,
            word_confidences=[WordConfidence("Test", 90, (0, 0, 30, 20))],
            character_confidences=[CharacterConfidence("T", 92, (0, 0, 10, 20))],
            lines=[
                LineData(
                    text="Test",
                    bbox=(0, 0, 30, 20),
                    confidence=90,
                    words=[],
                    baseline=20,
                )
            ],
            problematic_regions=[
                ProblematicRegion(
                    bbox=(0, 0, 30, 20), confidence=45, issue_type="low_confidence"
                )
            ],
            processing_time=1.0,
        )

        # Test to_dict
        dict_data = result.to_dict()
        assert dict_data["text"] == "Test"
        assert len(dict_data["word_confidences"]) == 1
        assert len(dict_data["problematic_regions"]) == 1

        # Test from_dict
        reconstructed = EnhancedOCRResult.from_dict(dict_data)
        assert reconstructed.text == result.text
        assert reconstructed.overall_confidence == result.overall_confidence
        assert len(reconstructed.word_confidences) == 1
