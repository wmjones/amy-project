"""
Test suite for AI-driven summarization and OCR-AI pipeline integration.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import time
from datetime import datetime

from src.metadata_extraction.ai_summarizer import (
    AISummarizer, 
    DocumentSummary
)
from src.metadata_extraction.ocr_ai_pipeline import (
    OCRAIPipeline,
    PipelineResult
)
from src.file_access.ocr_processor import OCRResult
from src.claude_integration.client import AnalysisResult


class TestAISummarizer(unittest.TestCase):
    """Test the AI summarizer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Claude client
        self.mock_claude = Mock()
        self.summarizer = AISummarizer(claude_client=self.mock_claude)
        
        # Sample test data
        self.test_file = Path("test_document.jpg")
        self.test_text = """
        Syracuse Salt Company
        Invoice #1234
        Date: March 15, 1892
        Location: Salina Street, Syracuse, New York
        """
    
    def test_initialization(self):
        """Test summarizer initialization."""
        self.assertIsNotNone(self.summarizer.claude_client)
        self.assertIsNotNone(self.summarizer.syracuse_landmarks)
        self.assertTrue(len(self.summarizer.historical_periods) > 0)
    
    def test_syracuse_knowledge_base(self):
        """Test Syracuse-specific knowledge base."""
        self.assertIn("Armory Square", self.summarizer.syracuse_landmarks)
        self.assertIn("salt_era", self.summarizer.historical_periods)
        self.assertIn("photo", self.summarizer.document_patterns)
    
    def test_summarize_document_success(self):
        """Test successful document summarization."""
        # Mock Claude response
        mock_result = AnalysisResult(
            content=json.dumps({
                "summary": "Historical invoice from Syracuse Salt Company",
                "category": "business_record",
                "confidence_score": 0.85,
                "key_entities": {
                    "organizations": ["Syracuse Salt Company"],
                    "locations": ["Syracuse", "Salina Street"],
                    "dates": ["1892-03-15"]
                },
                "historical_period": "salt_era",
                "content_type": "document",
                "suggested_folder_path": "Hansman_Syracuse/business_record/salt_era"
            }),
            metadata={},
            confidence_score=0.85,
            tokens_used=100,
            model="claude-3-7-sonnet"
        )
        
        self.mock_claude.analyze_document.return_value = mock_result
        
        # Test summarization
        summary = self.summarizer.summarize_document(
            file_path=self.test_file,
            ocr_text=self.test_text,
            ocr_confidence=0.9
        )
        
        # Assertions
        self.assertIsInstance(summary, DocumentSummary)
        self.assertEqual(summary.category, "business_record")
        self.assertEqual(summary.historical_period, "salt_era")
        self.assertIn("Syracuse Salt Company", summary.key_entities.get("organizations", []))
        self.assertGreater(summary.confidence_score, 0.8)
    
    def test_summarize_document_error(self):
        """Test error handling in summarization."""
        # Mock Claude error
        self.mock_claude.analyze_document.side_effect = Exception("API Error")
        
        # Test summarization
        summary = self.summarizer.summarize_document(
            file_path=self.test_file,
            ocr_text=self.test_text
        )
        
        # Assertions
        self.assertEqual(summary.category, "error")
        self.assertEqual(summary.confidence_score, 0.0)
        self.assertIsNotNone(summary.error_message)
        self.assertIn("API Error", summary.error_message)
    
    def test_parse_claude_response(self):
        """Test parsing various Claude response formats."""
        # Test valid JSON
        json_response = AnalysisResult(
            content='{"category": "photo", "summary": "Test"}',
            metadata={},
            confidence_score=0.8,
            tokens_used=50,
            model="claude-3-7-sonnet"
        )
        
        result = self.summarizer._parse_claude_response(json_response)
        self.assertEqual(result["category"], "photo")
        
        # Test embedded JSON
        embedded_response = AnalysisResult(
            content='Here is the analysis: {"category": "document"}',
            metadata={},
            confidence_score=0.8,
            tokens_used=50,
            model="claude-3-7-sonnet"
        )
        
        result = self.summarizer._parse_claude_response(embedded_response)
        self.assertEqual(result["category"], "document")
        
        # Test plain text
        text_response = AnalysisResult(
            content='This is a plain text response',
            metadata={},
            confidence_score=0.7,
            tokens_used=50,
            model="claude-3-7-sonnet"
        )
        
        result = self.summarizer._parse_claude_response(text_response)
        self.assertIn("summary", result)
        self.assertEqual(result["category"], "unknown")
    
    def test_enhance_with_local_knowledge(self):
        """Test enhancement with Syracuse-specific knowledge."""
        # Test data with Syracuse references
        data = {
            "summary": "Document about salt springs",
            "location_references": [],
            "date_references": ["1850"],
            "classification_tags": []
        }
        
        text = "This document discusses the Salt Springs of Syracuse"
        
        enhanced = self.summarizer._enhance_with_local_knowledge(data, text)
        
        self.assertIn("Salt Springs", enhanced["location_references"])
        self.assertIn("salt_industry", enhanced["classification_tags"])
        self.assertEqual(enhanced["historical_period"], "salt_era")
    
    def test_find_syracuse_references(self):
        """Test finding Syracuse-specific references."""
        text = """
        Visit Armory Square in downtown Syracuse.
        The Erie Canal runs through Central New York.
        Syracuse University is on University Hill.
        """
        
        references = self.summarizer._find_syracuse_references(text)
        
        self.assertIn("Armory Square", references)
        self.assertIn("Syracuse", references)
        self.assertIn("Central New York", references)
        self.assertIn("University Hill", references)
    
    def test_analyze_filename(self):
        """Test filename analysis."""
        # Test with date
        result = self.summarizer._analyze_filename("syracuse_1925_photo.jpg")
        self.assertIn("possible year", result)
        self.assertIn("Syracuse reference", result)
        self.assertIn("likely photograph", result)
        
        # Test with document type
        result = self.summarizer._analyze_filename("letter_001.pdf")
        self.assertIn("likely document", result)
        
        # Test with no patterns
        result = self.summarizer._analyze_filename("random.txt")
        self.assertEqual(result, "")
    
    def test_batch_summarize(self):
        """Test batch summarization."""
        # Mock documents
        documents = [
            (Path("doc1.jpg"), "Text 1", None),
            (Path("doc2.jpg"), "Text 2", {"extra": "context"}),
            (Path("doc3.jpg"), "Text 3", None)
        ]
        
        # Mock Claude responses
        self.mock_claude.analyze_document.return_value = AnalysisResult(
            content='{"category": "photo", "summary": "Test photo"}',
            metadata={},
            confidence_score=0.8,
            tokens_used=50,
            model="claude-3-7-sonnet"
        )
        
        # Progress callback
        progress_calls = []
        def progress_callback(done, total):
            progress_calls.append((done, total))
        
        # Test batch processing
        summaries = self.summarizer.batch_summarize(
            documents,
            batch_size=2,
            progress_callback=progress_callback
        )
        
        # Assertions
        self.assertEqual(len(summaries), 3)
        self.assertEqual(len(progress_calls), 2)  # Called twice for 3 docs with batch_size=2
        self.assertTrue(all(isinstance(s, DocumentSummary) for s in summaries))
    
    def test_create_collection_report(self):
        """Test collection report generation."""
        # Create sample summaries
        summaries = []
        for i in range(5):
            summary = DocumentSummary(
                file_path=f"file_{i}.jpg",
                ocr_text="Sample text",
                summary=f"Summary {i}",
                category="photo" if i % 2 == 0 else "document",
                confidence_score=0.7 + i * 0.05,
                key_entities={},
                date_references=[str(1900 + i * 10)],
                photo_subjects=[],
                location_references=["Syracuse"] if i % 2 == 0 else [],
                content_type="photo",
                historical_period="industrial_boom",
                classification_tags=[],
                claude_metadata={},
                processing_time=2.0 + i * 0.1
            )
            summaries.append(summary)
        
        # Generate report
        report = self.summarizer.create_collection_report(summaries)
        
        # Assertions
        self.assertEqual(report["total_documents"], 5)
        self.assertEqual(report["statistics"]["by_category"]["photo"], 3)
        self.assertEqual(report["statistics"]["by_category"]["document"], 2)
        self.assertIn("Syracuse", report["statistics"]["locations_mentioned"])
        self.assertGreater(report["statistics"]["confidence_distribution"]["medium"], 0)
        self.assertIn("Most mentioned location", report["key_findings"][0])


class TestOCRAIPipeline(unittest.TestCase):
    """Test the integrated OCR + AI pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock components
        self.mock_ocr = Mock()
        self.mock_ai = Mock()
        self.mock_claude = Mock()
        
        self.pipeline = OCRAIPipeline(
            ocr_processor=self.mock_ocr,
            ai_summarizer=self.mock_ai,
            claude_client=self.mock_claude,
            cache_results=False  # Disable caching for tests
        )
        
        self.test_file = Path("test.jpg")
    
    def test_process_file_success(self):
        """Test successful file processing."""
        # Mock OCR result
        ocr_result = OCRResult(
            text="Test OCR text",
            confidence=0.85,
            engine_used="tesseract",
            processing_time=1.5,
            page_count=1,
            pages=[],
            cached=False
        )
        self.mock_ocr.process_file.return_value = ocr_result
        
        # Mock AI summary
        ai_summary = DocumentSummary(
            file_path=str(self.test_file),
            ocr_text="Test OCR text",
            summary="Test summary",
            category="document",
            confidence_score=0.9,
            key_entities={},
            date_references=[],
            photo_subjects=[],
            location_references=[],
            content_type="document",
            historical_period="modern",
            classification_tags=[],
            claude_metadata={},
            processing_time=2.0
        )
        self.mock_ai.summarize_document.return_value = ai_summary
        
        # Test processing
        result = self.pipeline.process_file(self.test_file)
        
        # Assertions
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.success)
        self.assertEqual(result.ocr_result.text, "Test OCR text")
        self.assertEqual(result.ai_summary.category, "document")
        self.assertGreater(result.processing_time, 0)
    
    def test_process_file_ocr_error(self):
        """Test handling of OCR errors."""
        # Mock OCR error
        self.mock_ocr.process_file.side_effect = Exception("OCR failed")
        
        # Test processing
        result = self.pipeline.process_file(self.test_file)
        
        # Assertions
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("OCR failed", result.error_message)
    
    def test_process_batch(self):
        """Test batch processing."""
        # Mock successful processing
        ocr_result = OCRResult(
            text="Test text",
            confidence=0.8,
            engine_used="tesseract",
            processing_time=1.0
        )
        
        ai_summary = DocumentSummary(
            file_path="test.jpg",
            ocr_text="Test text",
            summary="Summary",
            category="photo",
            confidence_score=0.85,
            key_entities={},
            date_references=[],
            photo_subjects=[],
            location_references=[],
            content_type="photo",
            historical_period="modern",
            classification_tags=[],
            claude_metadata={},
            processing_time=1.5
        )
        
        self.mock_ocr.process_file.return_value = ocr_result
        self.mock_ai.summarize_document.return_value = ai_summary
        
        # Test batch
        files = [Path(f"test{i}.jpg") for i in range(3)]
        
        progress_calls = []
        def progress_callback(done, total):
            progress_calls.append((done, total))
        
        results = self.pipeline.process_batch(
            files,
            max_workers=2,
            progress_callback=progress_callback
        )
        
        # Assertions
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(len(progress_calls), 3)
    
    def test_caching(self):
        """Test result caching functionality."""
        # Enable caching for this test
        self.pipeline.cache_results = True
        
        # Mock successful processing
        ocr_result = OCRResult(
            text="Cached text",
            confidence=0.9,
            engine_used="tesseract",
            processing_time=1.0
        )
        
        ai_summary = DocumentSummary(
            file_path=str(self.test_file),
            ocr_text="Cached text",
            summary="Cached summary",
            category="document",
            confidence_score=0.95,
            key_entities={},
            date_references=[],
            photo_subjects=[],
            location_references=[],
            content_type="document",
            historical_period="modern",
            classification_tags=[],
            claude_metadata={},
            processing_time=1.5
        )
        
        self.mock_ocr.process_file.return_value = ocr_result
        self.mock_ai.summarize_document.return_value = ai_summary
        
        # First processing
        with patch.object(self.pipeline, '_calculate_file_hash', return_value='test_hash'):
            result1 = self.pipeline.process_file(self.test_file)
        
        # Verify not cached
        self.assertFalse(result1.cache_hit)
        
        # Second processing (should hit cache)
        with patch.object(self.pipeline, '_calculate_file_hash', return_value='test_hash'):
            with patch.object(self.pipeline, '_check_cache') as mock_check:
                mock_check.return_value = result1
                result2 = self.pipeline.process_file(self.test_file)
        
        # Verify cache hit
        self.assertTrue(result2.cache_hit)
        self.assertEqual(result2.ai_summary.summary, "Cached summary")
    
    def test_generate_pipeline_report(self):
        """Test pipeline report generation."""
        # Create sample results
        results = []
        
        for i in range(5):
            ocr_result = OCRResult(
                text=f"Text {i}",
                confidence=0.7 + i * 0.05,
                engine_used="tesseract" if i % 2 == 0 else "google",
                processing_time=1.0 + i * 0.1
            )
            
            ai_summary = DocumentSummary(
                file_path=f"file_{i}.jpg",
                ocr_text=f"Text {i}",
                summary=f"Summary {i}",
                category="photo" if i % 2 == 0 else "document",
                confidence_score=0.8 + i * 0.03,
                key_entities={},
                date_references=[],
                photo_subjects=[],
                location_references=[],
                content_type="photo",
                historical_period="modern",
                classification_tags=[],
                claude_metadata={},
                processing_time=2.0 + i * 0.2
            )
            
            pipeline_result = PipelineResult(
                file_path=Path(f"file_{i}.jpg"),
                ocr_result=ocr_result,
                ai_summary=ai_summary,
                processing_time=3.0 + i * 0.3,
                cache_hit=i % 3 == 0,
                success=i < 4,  # Last one fails
                error_message="Test error" if i == 4 else None
            )
            
            results.append(pipeline_result)
        
        # Generate report
        report = self.pipeline.generate_pipeline_report(results)
        
        # Assertions
        self.assertEqual(report["total_files"], 5)
        self.assertEqual(report["statistics"]["successful"], 4)
        self.assertEqual(report["statistics"]["failed"], 1)
        self.assertIn("tesseract", report["statistics"]["ocr_engines"])
        self.assertIn("google", report["statistics"]["ocr_engines"])
        self.assertEqual(len(report["errors"]), 1)
        self.assertGreater(report["statistics"]["total_processing_time"], 0)
    
    def test_export_results(self):
        """Test result export functionality."""
        # Create sample result
        result = PipelineResult(
            file_path=Path("test.jpg"),
            ocr_result=OCRResult(
                text="Export test",
                confidence=0.85,
                engine_used="tesseract",
                processing_time=1.0
            ),
            ai_summary=DocumentSummary(
                file_path="test.jpg",
                ocr_text="Export test",
                summary="Test export summary",
                category="document",
                confidence_score=0.9,
                key_entities={},
                date_references=[],
                photo_subjects=[],
                location_references=[],
                content_type="document",
                historical_period="modern",
                classification_tags=[],
                claude_metadata={},
                processing_time=2.0
            ),
            processing_time=3.0,
            success=True
        )
        
        results = [result]
        
        # Test JSON export
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            json_path = self.pipeline.export_results(results, format="json")
            self.assertTrue(str(json_path).endswith(".json"))
        
        # Test CSV export
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            csv_path = self.pipeline.export_results(results, format="csv")
            self.assertTrue(str(csv_path).endswith(".csv"))
        
        # Test summary export
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            summary_path = self.pipeline.export_results(results, format="summary")
            self.assertTrue(str(summary_path).endswith(".txt"))


if __name__ == "__main__":
    unittest.main()