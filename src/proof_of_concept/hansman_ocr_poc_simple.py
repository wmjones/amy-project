#!/usr/bin/env python3
"""
Simplified OCR Proof of Concept for Hansman Syracuse Collection.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HansmanOCRPOC:
    """Simplified OCR proof of concept for Hansman Syracuse Collection."""
    
    def __init__(self):
        """Initialize the POC."""
        self.setup_directories()
        self.metrics = {
            'total_files': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'processing_times': {},
            'ocr_results': {}
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            'data/hansman_samples',
            'output/ocr_results',
            'output/reports',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_sample_files(self, sample_dir: str = 'data/hansman_samples') -> List[Path]:
        """Get sample files."""
        sample_path = Path(sample_dir)
        
        if not sample_path.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return []
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        files = []
        
        for ext in image_extensions:
            files.extend(sample_path.glob(f'*{ext}'))
            files.extend(sample_path.glob(f'*{ext.upper()}'))
        
        files = sorted(files)
        logger.info(f"Found {len(files)} sample files to process")
        return files
    
    def preprocess_image(self, image_path: Path) -> Image.Image:
        """Preprocess image for better OCR."""
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Apply sharpening
        img = img.filter(ImageFilter.SHARPEN)
        
        # Convert to numpy array for additional processing
        img_array = np.array(img)
        
        # Apply thresholding
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_array)
        
        return img
    
    def extract_text(self, image_path: Path) -> Dict[str, Any]:
        """Extract text from image using Tesseract."""
        start_time = time.time()
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_img)
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'text': text.strip(),
                'confidence': avg_confidence,
                'processing_time': processing_time,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_all_samples(self) -> Dict[str, Any]:
        """Process all sample files."""
        samples = self.get_sample_files()
        
        if not samples:
            logger.error("No sample files found")
            return self.metrics
        
        self.metrics['total_files'] = len(samples)
        
        for sample_file in samples:
            logger.info(f"Processing: {sample_file.name}")
            
            # Extract text
            result = self.extract_text(sample_file)
            
            if result['success']:
                self.metrics['successful_ocr'] += 1
                self.metrics['ocr_results'][sample_file.name] = result
                
                # Save OCR result
                output_file = Path('output/ocr_results') / f"{sample_file.stem}_ocr.txt"
                output_file.write_text(result['text'])
                
                logger.info(f"✓ OCR completed for {sample_file.name} "
                           f"(confidence: {result['confidence']:.1f}%, "
                           f"time: {result['processing_time']:.2f}s)")
            else:
                self.metrics['failed_ocr'] += 1
                logger.error(f"✗ OCR failed for {sample_file.name}: {result['error']}")
            
            self.metrics['processing_times'][sample_file.name] = result['processing_time']
        
        return self.metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': self.metrics['total_files'],
                'successful_ocr': self.metrics['successful_ocr'],
                'failed_ocr': self.metrics['failed_ocr'],
                'success_rate': (self.metrics['successful_ocr'] / self.metrics['total_files'] * 100) if self.metrics['total_files'] > 0 else 0
            },
            'performance': {
                'average_processing_time': np.mean(list(self.metrics['processing_times'].values())) if self.metrics['processing_times'] else 0,
                'total_processing_time': sum(self.metrics['processing_times'].values()),
                'processing_speed': self.metrics['total_files'] / sum(self.metrics['processing_times'].values()) if sum(self.metrics['processing_times'].values()) > 0 else 0
            },
            'ocr_results': self.metrics['ocr_results'],
            'cost_estimates': self.calculate_cost_estimates(),
            'recommendations': self.generate_recommendations()
        }
        
        # Save JSON report
        report_path = Path('output/reports/hansman_poc_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        return report
    
    def calculate_cost_estimates(self) -> Dict[str, Any]:
        """Calculate cost estimates for processing the full collection."""
        collection_size = 400  # Estimated number of documents
        
        avg_time = np.mean(list(self.metrics['processing_times'].values())) if self.metrics['processing_times'] else 2.0
        total_time_hours = (collection_size * avg_time) / 3600
        
        # Simplified cost calculation
        return {
            'collection_size': collection_size,
            'average_processing_time_per_doc': avg_time,
            'total_processing_time_hours': total_time_hours,
            'compute_cost_estimate': f"${total_time_hours * 0.10:.2f}",  # Assuming $0.10/hour compute
            'note': "This estimate includes only compute costs. Claude API costs would be additional."
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on POC results."""
        recommendations = []
        
        if self.metrics['successful_ocr'] == self.metrics['total_files']:
            recommendations.append("OCR processing is highly reliable with current settings")
        elif self.metrics['successful_ocr'] / self.metrics['total_files'] > 0.8:
            recommendations.append("OCR processing shows good reliability but may need fine-tuning for some documents")
        else:
            recommendations.append("OCR processing needs significant optimization for this collection")
        
        avg_confidence = np.mean([r['confidence'] for r in self.metrics['ocr_results'].values() if 'confidence' in r])
        if avg_confidence > 90:
            recommendations.append("Text extraction quality is excellent")
        elif avg_confidence > 70:
            recommendations.append("Text extraction quality is good but could benefit from image preprocessing improvements")
        else:
            recommendations.append("Consider enhanced image preprocessing to improve OCR accuracy")
        
        recommendations.append("Implement batch processing for optimal throughput")
        recommendations.append("Set up automated quality checks for OCR results")
        
        return recommendations
    
    def generate_markdown_report(self, report: Dict[str, Any]):
        """Generate a markdown report."""
        md_content = f"""# Hansman Syracuse Collection OCR Proof of Concept Report

Generated: {report['timestamp']}

## Summary

- **Total Files Processed**: {report['summary']['total_files']}
- **Successful OCR**: {report['summary']['successful_ocr']}
- **Failed OCR**: {report['summary']['failed_ocr']}
- **Success Rate**: {report['summary']['success_rate']:.1f}%

## Performance Metrics

- **Average Processing Time**: {report['performance']['average_processing_time']:.2f} seconds/document
- **Total Processing Time**: {report['performance']['total_processing_time']:.2f} seconds
- **Processing Speed**: {report['performance']['processing_speed']:.2f} documents/second

## OCR Results

"""
        
        for filename, result in report['ocr_results'].items():
            if result.get('success'):
                md_content += f"### {filename}\n"
                md_content += f"- **Confidence**: {result['confidence']:.1f}%\n"
                md_content += f"- **Processing Time**: {result['processing_time']:.2f}s\n"
                md_content += f"- **Word Count**: {result['word_count']}\n"
                md_content += f"- **Character Count**: {result['char_count']}\n\n"
        
        md_content += f"""## Cost Estimates

- **Collection Size**: {report['cost_estimates']['collection_size']} documents
- **Average Processing Time**: {report['cost_estimates']['average_processing_time_per_doc']:.2f} seconds/document
- **Total Processing Time**: {report['cost_estimates']['total_processing_time_hours']:.2f} hours
- **Estimated Compute Cost**: {report['cost_estimates']['compute_cost_estimate']}

*Note: {report['cost_estimates']['note']}*

## Recommendations

"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        # Save markdown report
        md_path = Path('output/reports/hansman_poc_report.md')
        md_path.write_text(md_content)


def main():
    """Run the proof of concept."""
    poc = HansmanOCRPOC()
    
    logger.info("Starting Hansman Syracuse Collection OCR Proof of Concept")
    
    # Process all samples
    poc.process_all_samples()
    
    # Generate report
    report = poc.generate_report()
    
    # Print summary
    print("\n=== POC Results ===")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average Processing Time: {report['performance']['average_processing_time']:.2f}s")
    print(f"Reports generated in output/reports/")


if __name__ == "__main__":
    main()