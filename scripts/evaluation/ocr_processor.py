#!/usr/bin/env python3
"""OCR Processing and Text Extraction Module for the Hansman Syracuse Collection."""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytesseract
from PIL import Image
import cv2
import numpy as np
from google.cloud import vision
import redis
from tenacity import retry, stop_after_attempt, wait_exponential
import PyPDF2
from pdf2image import convert_from_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Container for OCR processing results."""
    file_id: str
    text: str
    confidence: float
    language: str
    processing_time: float
    ocr_engine: str
    page_count: int
    errors: List[str]
    preprocessed: bool
    cached: bool
    timestamp: datetime

class OCRCache:
    """Redis-based caching for OCR results."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 expire_time: int = 86400):  # 24 hours default
        self.expire_time = expire_time
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
            self.enabled = True
            logger.info("Redis cache connected")
        except:
            self.redis_client = None
            self.enabled = False
            logger.warning("Redis not available, caching disabled")
    
    def get(self, file_hash: str) -> Optional[Dict]:
        """Retrieve cached OCR result."""
        if not self.enabled:
            return None
        
        try:
            data = self.redis_client.get(f"ocr:{file_hash}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, file_hash: str, result: Dict):
        """Store OCR result in cache."""
        if not self.enabled:
            return
        
        try:
            self.redis_client.setex(
                f"ocr:{file_hash}",
                self.expire_time,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")

class OCRProcessor:
    """Main OCR processing engine with fallback mechanisms."""
    
    def __init__(self, 
                 tesseract_config: Optional[Dict] = None,
                 google_credentials: Optional[str] = None,
                 enable_preprocessing: bool = True,
                 cache_enabled: bool = True):
        
        # Configure Tesseract
        self.tesseract_config = tesseract_config or {
            'lang': 'eng',
            'config': '--oem 3 --psm 6',  # LSTM engine, uniform text block
            'timeout': 30
        }
        
        # Configure Google Cloud Vision (if credentials provided)
        self.google_client = None
        if google_credentials and os.path.exists(google_credentials):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials
            try:
                self.google_client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Vision: {e}")
        
        # Configuration
        self.enable_preprocessing = enable_preprocessing
        self.confidence_threshold = 0.6
        self.max_retries = 3
        
        # Initialize cache
        self.cache = OCRCache() if cache_enabled else None
        
        # Initialize preprocessor
        if enable_preprocessing:
            from ocr_preprocessing import DocumentPreprocessor
            self.preprocessor = DocumentPreprocessor()
        else:
            self.preprocessor = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'tesseract_success': 0,
            'google_success': 0,
            'cache_hits': 0,
            'failures': 0
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for caching."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _tesseract_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Perform OCR using Tesseract."""
        try:
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                lang=self.tesseract_config['lang'],
                config=self.tesseract_config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self.tesseract_config['lang'],
                config=self.tesseract_config['config']
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text.strip(), avg_confidence / 100
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _google_vision_ocr(self, image_path: Path) -> Tuple[str, float]:
        """Perform OCR using Google Cloud Vision."""
        if not self.google_client:
            raise ValueError("Google Cloud Vision client not initialized")
        
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform text detection
            response = self.google_client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # Full text is in the first annotation
                full_text = texts[0].description
                
                # Calculate confidence (Google provides per-word confidence)
                word_confidences = []
                for text in texts[1:]:  # Skip the first full text annotation
                    if hasattr(text, 'confidence'):
                        word_confidences.append(text.confidence)
                
                avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.9
                
                return full_text.strip(), avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Google Vision OCR error: {e}")
            raise
    
    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess image for better OCR results."""
        if not self.preprocessor:
            return cv2.imread(str(image_path))
        
        try:
            # Use the preprocessing pipeline
            results = self.preprocessor.process_image(image_path)
            return results['final']
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Return original image if preprocessing fails
            return cv2.imread(str(image_path))
    
    def _process_pdf(self, pdf_path: Path) -> List[Tuple[np.ndarray, int]]:
        """Convert PDF pages to images for OCR."""
        images = []
        
        try:
            # Convert PDF to images
            pdf_images = convert_from_path(pdf_path, dpi=300)
            
            for page_num, pil_image in enumerate(pdf_images, 1):
                # Convert PIL Image to numpy array
                numpy_image = np.array(pil_image)
                # Convert RGB to BGR for OpenCV
                if len(numpy_image.shape) == 3:
                    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                
                images.append((numpy_image, page_num))
                
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            # Try alternative method using PyPDF2 for text extraction
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            # Create a simple text image for consistency
                            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                            images.append((img, page_num))
            except:
                pass
        
        return images
    
    def process_file(self, file_path: Path, file_metadata: Optional[Dict] = None) -> OCRResult:
        """Process a single file through the OCR pipeline."""
        start_time = time.time()
        file_hash = self._calculate_file_hash(file_path)
        file_id = file_metadata.get('file_id', file_hash) if file_metadata else file_hash
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(file_hash)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for {file_path.name}")
                return OCRResult(
                    file_id=file_id,
                    text=cached_result['text'],
                    confidence=cached_result['confidence'],
                    language=cached_result.get('language', 'eng'),
                    processing_time=0.0,
                    ocr_engine=cached_result.get('ocr_engine', 'cached'),
                    page_count=cached_result.get('page_count', 1),
                    errors=[],
                    preprocessed=cached_result.get('preprocessed', False),
                    cached=True,
                    timestamp=datetime.fromisoformat(cached_result['timestamp'])
                )
        
        # Initialize result
        all_text = []
        all_confidence = []
        errors = []
        page_count = 1
        ocr_engine = "none"
        preprocessed = False
        
        try:
            # Handle different file types
            if file_path.suffix.lower() == '.pdf':
                images = self._process_pdf(file_path)
                page_count = len(images)
            else:
                # Load and preprocess image
                if self.enable_preprocessing:
                    image = self._preprocess_image(file_path)
                    preprocessed = True
                else:
                    image = cv2.imread(str(file_path))
                
                images = [(image, 1)]
            
            # Process each page/image
            for image, page_num in images:
                page_text = ""
                page_confidence = 0.0
                
                # Try Tesseract first
                try:
                    text, confidence = self._tesseract_ocr(image)
                    
                    if confidence >= self.confidence_threshold:
                        page_text = text
                        page_confidence = confidence
                        ocr_engine = "tesseract"
                        self.stats['tesseract_success'] += 1
                        logger.info(f"Tesseract success for {file_path.name} page {page_num} (confidence: {confidence:.2f})")
                    else:
                        logger.warning(f"Tesseract confidence too low ({confidence:.2f}) for {file_path.name} page {page_num}")
                        
                except Exception as e:
                    logger.error(f"Tesseract failed for {file_path.name} page {page_num}: {e}")
                    errors.append(f"Tesseract error page {page_num}: {str(e)}")
                
                # Fallback to Google Vision if available and needed
                if page_confidence < self.confidence_threshold and self.google_client:
                    try:
                        # Save preprocessed image temporarily for Google Vision
                        temp_path = file_path.parent / f"temp_{file_path.stem}_page{page_num}.png"
                        cv2.imwrite(str(temp_path), image)
                        
                        text, confidence = self._google_vision_ocr(temp_path)
                        
                        if confidence > page_confidence:
                            page_text = text
                            page_confidence = confidence
                            ocr_engine = "google_vision"
                            self.stats['google_success'] += 1
                            logger.info(f"Google Vision success for {file_path.name} page {page_num} (confidence: {confidence:.2f})")
                        
                        # Clean up temp file
                        temp_path.unlink()
                        
                    except Exception as e:
                        logger.error(f"Google Vision failed for {file_path.name} page {page_num}: {e}")
                        errors.append(f"Google Vision error page {page_num}: {str(e)}")
                
                all_text.append(page_text)
                all_confidence.append(page_confidence)
            
            # Combine results
            final_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text) if page_count > 1 else all_text[0]
            final_confidence = sum(all_confidence) / len(all_confidence) if all_confidence else 0.0
            
            # Create result
            result = OCRResult(
                file_id=file_id,
                text=final_text,
                confidence=final_confidence,
                language=self.tesseract_config['lang'],
                processing_time=time.time() - start_time,
                ocr_engine=ocr_engine,
                page_count=page_count,
                errors=errors,
                preprocessed=preprocessed,
                cached=False,
                timestamp=datetime.now()
            )
            
            # Cache the result
            if self.cache and final_text:
                self.cache.set(file_hash, asdict(result))
            
            self.stats['total_processed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error processing {file_path.name}: {e}")
            self.stats['failures'] += 1
            
            return OCRResult(
                file_id=file_id,
                text="",
                confidence=0.0,
                language=self.tesseract_config['lang'],
                processing_time=time.time() - start_time,
                ocr_engine="error",
                page_count=0,
                errors=[f"Critical error: {str(e)}"],
                preprocessed=preprocessed,
                cached=False,
                timestamp=datetime.now()
            )
    
    def process_batch(self, file_paths: List[Path], 
                     max_workers: int = 4) -> Dict[str, OCRResult]:
        """Process multiple files in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_file, path): path 
                for path in file_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[str(path)] = result
                except Exception as e:
                    logger.error(f"Batch processing error for {path}: {e}")
                    results[str(path)] = OCRResult(
                        file_id=str(path),
                        text="",
                        confidence=0.0,
                        language="eng",
                        processing_time=0.0,
                        ocr_engine="error",
                        page_count=0,
                        errors=[f"Batch processing error: {str(e)}"],
                        preprocessed=False,
                        cached=False,
                        timestamp=datetime.now()
                    )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.stats['total_processed']
        
        return {
            'total_processed': total,
            'tesseract_success': self.stats['tesseract_success'],
            'google_success': self.stats['google_success'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / total if total > 0 else 0,
            'failures': self.stats['failures'],
            'failure_rate': self.stats['failures'] / total if total > 0 else 0,
            'tesseract_usage': self.stats['tesseract_success'] / total if total > 0 else 0,
            'google_usage': self.stats['google_success'] / total if total > 0 else 0
        }


def test_ocr_processor():
    """Test the OCR processor on sample files."""
    
    # Initialize processor
    processor = OCRProcessor(
        tesseract_config={
            'lang': 'eng',
            'config': '--oem 3 --psm 6'
        },
        google_credentials=None,  # Add path to credentials if available
        enable_preprocessing=True,
        cache_enabled=False  # Disable for testing
    )
    
    # Test on sample files
    test_dir = Path("/workspaces/amy-project/hansman_organized/downloads")
    sample_files = list(test_dir.glob("*.JPG"))[:5]
    
    print("Testing OCR Processor on Hansman Syracuse photos...")
    print("-" * 50)
    
    results = []
    
    for file_path in sample_files:
        print(f"\nProcessing: {file_path.name}")
        result = processor.process_file(file_path)
        results.append(result)
        
        print(f"  Engine: {result.ocr_engine}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Text length: {len(result.text)} chars")
        print(f"  Preprocessed: {result.preprocessed}")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        if result.text:
            preview = result.text[:100].replace('\n', ' ')
            print(f"  Preview: {preview}...")
    
    # Get statistics
    stats = processor.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch_results = processor.process_batch(sample_files[:3], max_workers=2)
    
    print(f"Batch processed {len(batch_results)} files")
    
    return processor, results


if __name__ == "__main__":
    test_ocr_processor()