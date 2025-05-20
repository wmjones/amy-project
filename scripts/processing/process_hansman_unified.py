#!/usr/bin/env python3
"""Unified processing script for Hansman Syracuse photo collection with step checking and force options.

Usage examples:
    # Basic processing from Dropbox (with smart skipping of completed steps)
    python process_hansman_unified.py
    
    # Process local files instead of Dropbox
    python process_hansman_unified.py --use-local --local-dir /workspaces/amy-project/workspace/downloads/hansman
    
    # Process only first 10 files for testing
    python process_hansman_unified.py --limit 10
    
    # Force re-download all files
    python process_hansman_unified.py --force-download
    
    # Force re-run OCR on all files
    python process_hansman_unified.py --force-ocr
    
    # Force re-run everything
    python process_hansman_unified.py --force-all
    
    # Use custom output directory
    python process_hansman_unified.py --output-dir ./my_hansman_results
    
    # Process local files with custom output
    python process_hansman_unified.py --use-local --local-dir ./hansman_files --output-dir ./results
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dropbox
from dotenv import load_dotenv

# Import enhanced OCR processor from preprocessing module
from src.preprocessing.ocr_enhanced import EnhancedOCRProcessor, EnhancedOCRResult
from src.preprocessing import PreprocessingPipeline, ImageEnhancer

# Import other components
from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
from src.claude_integration.client import ClaudeClient
from src.utils.report_generator import ReportGenerator
from src.organization_logic.engine import OrganizationEngine
from src.file_access.processor import FileProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hansman_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HansmanUnifiedProcessor:
    """Unified processing pipeline for Hansman collection with resume capability."""
    
    def __init__(self, output_dir: str = "./hansman_results", 
                 download_limit: Optional[int] = None,
                 force_redownload: bool = False,
                 force_reocr: bool = False,
                 force_reai: bool = False,
                 force_reorganize: bool = False,
                 use_local_files: bool = False,
                 local_dir: Optional[str] = None):
        """Initialize the unified processor.
        
        Args:
            output_dir: Directory to save all outputs
            download_limit: Optional limit on number of files to download (None for all)
            force_redownload: Force re-download of files even if they exist
            force_reocr: Force re-run OCR on all files
            force_reai: Force re-run AI summarization on all files
            force_reorganize: Force re-organize all files
            use_local_files: Use local files instead of Dropbox
            local_dir: Directory containing local files when use_local_files is True
        """
        self.output_dir = Path(output_dir)
        self.download_limit = download_limit
        self.force_redownload = force_redownload
        self.force_reocr = force_reocr
        self.force_reai = force_reai
        self.force_reorganize = force_reorganize
        self.use_local_files = use_local_files
        self.local_dir = Path(local_dir) if local_dir else None
        
        # Create output directories
        self.downloads_dir = self.output_dir / "downloads"
        self.ocr_results_dir = self.output_dir / "ocr_results"
        self.summaries_dir = self.output_dir / "summaries"
        self.reports_dir = self.output_dir / "reports"
        self.organized_dir = self.output_dir / "organized"
        self.state_dir = self.output_dir / ".state"
        
        for dir_path in [self.downloads_dir, self.ocr_results_dir, 
                        self.summaries_dir, self.reports_dir, 
                        self.organized_dir, self.state_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dbx = None
        self.ocr_processor = None
        self.ai_summarizer = None
        self.organization_engine = None
        
        # State tracking
        self.state_file = self.state_dir / "processing_state.json"
        self.state = self.load_state()
        
        # Results tracking
        self.results = []
        self.errors = []
    
    def load_state(self) -> Dict[str, Any]:
        """Load processing state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading state: {e}, starting fresh")
        
        return {
            'downloaded_files': {},  # filename -> metadata
            'ocr_processed': {},     # filename -> ocr_result_path
            'ai_processed': {},      # filename -> summary_path
            'organized_files': {},   # filename -> target_path
            'errors': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def save_state(self):
        """Save processing state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def file_needs_processing(self, filename: str, stage: str) -> bool:
        """Check if a file needs processing at a given stage."""
        if stage == 'download':
            if self.force_redownload:
                return True
            return filename not in self.state['downloaded_files']
        
        elif stage == 'ocr':
            if self.force_reocr:
                return True
            return filename not in self.state['ocr_processed']
        
        elif stage == 'ai':
            if self.force_reai:
                return True
            return filename not in self.state['ai_processed']
        
        elif stage == 'organize':
            if self.force_reorganize:
                return True
            return filename not in self.state['organized_files']
        
        return True
    
    def connect_to_dropbox(self) -> bool:
        """Connect to Dropbox using credentials from environment."""
        try:
            logger.info("Connecting to Dropbox...")
            self.dbx = dropbox.Dropbox(
                app_key=os.getenv('DROPBOX_APP_KEY'),
                app_secret=os.getenv('DROPBOX_APP_SECRET'),
                oauth2_access_token=os.getenv('DROPBOX_ACCESS_TOKEN')
            )
            
            # Verify connection
            account = self.dbx.users_get_current_account()
            logger.info(f"Connected to Dropbox as: {account.email}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Dropbox: {e}")
            self.errors.append({"stage": "dropbox_connection", "error": str(e)})
            return False
    
    def list_files(self, folder_path: str = "/Hansman Syracuse photo docs July 2015") -> List[dropbox.files.FileMetadata]:
        """List all files in Dropbox folder."""
        if not self.dbx:
            logger.error("Dropbox not connected")
            return []
        
        all_files = []
        try:
            logger.info(f"Listing files in: {folder_path}")
            result = self.dbx.files_list_folder(folder_path)
            
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        # Only process supported file types
                        if entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', 
                                                       '.pdf', '.docx', '.doc', '.txt')):
                            all_files.append(entry)
                
                if not result.has_more:
                    break
                result = self.dbx.files_list_folder_continue(result.cursor)
            
            logger.info(f"Found {len(all_files)} supported files")
            return all_files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            self.errors.append({"stage": "list_files", "error": str(e)})
            return []
    
    def list_local_files(self) -> List[Path]:
        """List all supported files in local directory."""
        if not self.local_dir:
            logger.error("Local directory not specified")
            return []
        
        all_files = []
        supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', 
                               '.pdf', '.docx', '.doc', '.txt')
        
        logger.info(f"Scanning local directory: {self.local_dir}")
        
        for file_path in self.local_dir.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} supported files in local directory")
        return sorted(all_files)  # Sort for consistent processing order
    
    def download_files(self, files: List[dropbox.files.FileMetadata]) -> Dict[str, Any]:
        """Download files from Dropbox with progress tracking."""
        download_stats = {
            'downloaded': 0,
            'skipped': 0,
            'errors': 0,
            'files': []
        }
        
        # Apply download limit if specified
        if self.download_limit:
            files = files[:self.download_limit]
            logger.info(f"Limited to {self.download_limit} files")
        
        logger.info(f"Processing {len(files)} files for download...")
        
        # Check which files need downloading
        files_to_download = []
        for file_entry in files:
            if self.file_needs_processing(file_entry.name, 'download'):
                files_to_download.append(file_entry)
            else:
                download_stats['skipped'] += 1
                logger.info(f"Skipping already downloaded: {file_entry.name}")
        
        # Download needed files
        if files_to_download:
            for file_entry in tqdm(files_to_download, desc="Downloading files"):
                try:
                    local_path = self.downloads_dir / file_entry.name
                    
                    # Download file
                    _, response = self.dbx.files_download(file_entry.path_display)
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Update state
                    self.state['downloaded_files'][file_entry.name] = {
                        "path": str(local_path),
                        "size": file_entry.size,
                        "modified": str(file_entry.client_modified),
                        "downloaded_at": datetime.now().isoformat()
                    }
                    
                    download_stats['downloaded'] += 1
                    download_stats['files'].append(file_entry.name)
                    
                except Exception as e:
                    logger.error(f"Error downloading {file_entry.name}: {e}")
                    self.errors.append({
                        "stage": "download",
                        "file": file_entry.name,
                        "error": str(e)
                    })
                    download_stats['errors'] += 1
        
        self.save_state()
        return download_stats
    
    def initialize_processors(self) -> bool:
        """Initialize OCR and AI processors."""
        try:
            logger.info("Initializing processors...")
            
            # Initialize preprocessing pipeline
            self.preprocessing_pipeline = PreprocessingPipeline()
            
            # Initialize enhanced OCR processor with caching
            self.ocr_processor = EnhancedOCRProcessor(
                language='eng',
                confidence_threshold=60.0,
                parallel_processing=True,
                max_workers=4,
                cache_enabled=True,
                cache_dir=str(self.state_dir / 'ocr_cache')
            )
            
            # Initialize AI summarizer
            claude_client = ClaudeClient(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.ai_summarizer = AISummarizer(claude_client=claude_client)
            
            # Initialize organization engine with custom rules
            self.organization_engine = OrganizationEngine(
                rules_path=None,
                use_default_rules=True,
                default_folder="Unsorted"
            )
            
            # Add Hansman-specific organization rules
            self.add_hansman_organization_rules()
            
            logger.info("All processors initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            self.errors.append({"stage": "initialization", "error": str(e)})
            return False
    
    def add_hansman_organization_rules(self):
        """Add custom organization rules for Hansman collection."""
        # Rule for Syracuse lists
        self.organization_engine.add_rule({
            "name": "Syracuse Lists",
            "conditions": {
                "keywords": ["syracuse list", "syracuse directory"]
            },
            "path_template": "Syracuse_Lists/{filename}",
            "priority": 100
        })
        
        # Rule for dated photos
        self.organization_engine.add_rule({
            "name": "Dated Photos",
            "conditions": {
                "file_pattern": r"(\d{4})_",
                "document_type": "photo"
            },
            "path_template": "Photos/{dates.year}/{filename}",
            "priority": 90
        })
        
        # Rule for sequential photos
        self.organization_engine.add_rule({
            "name": "Sequential Photos",
            "conditions": {
                "file_pattern": r"^100_\d+"
            },
            "path_template": "Sequential_Photos/{filename}",
            "priority": 85
        })
        
        # Rule for correspondence
        self.organization_engine.add_rule({
            "name": "Correspondence",
            "conditions": {
                "keywords": ["arnold", "knowles", "letter", "correspondence"]
            },
            "path_template": "Correspondence/{filename}",
            "priority": 80
        })
    
    def process_ocr(self, filename: str) -> Optional[Dict[str, Any]]:
        """Process a single file through OCR using enhanced preprocessing and OCR."""
        if not self.file_needs_processing(filename, 'ocr'):
            logger.info(f"Skipping OCR for already processed: {filename}")
            return self.state['ocr_processed'].get(filename)
        
        try:
            # Get file path - either from local directory or downloads
            if self.use_local_files:
                local_path = Path(self.state['downloaded_files'][filename]['path'])
            else:
                local_path = self.downloads_dir / filename
                
            if not local_path.exists():
                logger.error(f"File not found for OCR: {filename}")
                return None
            
            logger.info(f"Processing OCR for: {filename}")
            
            # First, run preprocessing pipeline to get enhanced image variants
            preprocessing_result = self.preprocessing_pipeline.process_document(
                str(local_path),
                str(self.ocr_results_dir / local_path.stem)
            )
            
            # Get the enhanced image variants
            enhanced_images_paths = preprocessing_result.get('enhanced_images', {})
            metadata = preprocessing_result.get('metadata', None)
            
            # Load the enhanced images as numpy arrays
            enhanced_images = {}
            image_enhancer = self.preprocessing_pipeline.enhancer
            
            # Get the enhanced images directly from the enhancer
            enhanced_images = image_enhancer.enhance_image(str(local_path))
            
            # Process variants through enhanced OCR
            ocr_result = self.ocr_processor.process_variants(
                enhanced_images,
                str(local_path)
            )
            
            if ocr_result:
                # Save OCR text
                ocr_file = self.ocr_results_dir / f"{local_path.stem}_ocr.txt"
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    f.write(ocr_result.text)
                
                # Save enhanced OCR result as JSON
                ocr_json_file = self.ocr_results_dir / f"{local_path.stem}_ocr.json"
                result_data = {
                    "filename": filename,
                    "text": ocr_result.text,
                    "overall_confidence": ocr_result.overall_confidence,
                    "processing_time": ocr_result.processing_time,
                    "word_count": len(ocr_result.word_confidences),
                    "line_count": len(ocr_result.lines),
                    "problematic_regions": [
                        {
                            "bbox": region.bbox,
                            "confidence": region.confidence,
                            "issue_type": region.issue_type,
                            "suggested_action": region.suggested_action
                        }
                        for region in ocr_result.problematic_regions
                    ],
                    "variant_results": ocr_result.variant_results,
                    "preprocessing_metadata": metadata.to_dict() if metadata else None,
                    "processed_at": datetime.now().isoformat()
                }
                
                # Custom JSON encoder for numpy types
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if hasattr(obj, 'item'):  # numpy scalar
                            return obj.item()
                        if hasattr(obj, 'tolist'):  # numpy array
                            return obj.tolist()
                        return super().default(obj)
                
                with open(ocr_json_file, 'w') as f:
                    json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                
                # Export structured OCR output
                ocr_export_file = self.ocr_results_dir / f"{local_path.stem}_ocr_structured.json"
                self.ocr_processor.export_structured_output(
                    ocr_result,
                    str(ocr_export_file),
                    format='json'
                )
                
                # Update state
                self.state['ocr_processed'][filename] = str(ocr_json_file)
                self.save_state()
                
                return result_data
            
        except Exception as e:
            logger.error(f"Error in OCR processing for {filename}: {e}")
            self.errors.append({
                "stage": "ocr",
                "file": filename,
                "error": str(e)
            })
        
        return None
    
    def process_ai_summary(self, filename: str, ocr_text: str) -> Optional[Dict[str, Any]]:
        """Process a file through AI summarization."""
        if not self.file_needs_processing(filename, 'ai'):
            logger.info(f"Skipping AI for already processed: {filename}")
            return self.state['ai_processed'].get(filename)
        
        try:
            logger.info(f"Processing AI summary for: {filename}")
            
            # Use the AI summarizer's document method
            from pathlib import Path
            # Get file path - either from local directory or downloads
            if self.use_local_files:
                file_path = Path(self.state['downloaded_files'][filename]['path'])
            else:
                file_path = self.downloads_dir / filename
            
            summary_result = self.ai_summarizer.summarize_document(
                ocr_text=ocr_text,
                file_path=file_path,
                additional_context={
                    'collection': 'Hansman Syracuse photo docs July 2015',
                    'processing_date': datetime.now().isoformat()
                }
            )
            
            if summary_result:
                # Save AI summary as structured data
                summary_file = self.summaries_dir / f"{Path(filename).stem}_summary.json"
                
                # Convert DocumentSummary dataclass to dict
                from dataclasses import asdict
                summary_data = asdict(summary_result)
                
                # Convert datetime objects to strings
                if 'created_at' in summary_data and hasattr(summary_data['created_at'], 'isoformat'):
                    summary_data['created_at'] = summary_data['created_at'].isoformat()
                
                summary_data['processed_at'] = datetime.now().isoformat()
                
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)  # Use default=str for any remaining datetime objects
                
                # Update state
                self.state['ai_processed'][filename] = str(summary_file)
                self.save_state()
                
                return summary_data
            
        except Exception as e:
            logger.error(f"Error in AI processing for {filename}: {e}")
            self.errors.append({
                "stage": "ai",
                "file": filename,
                "error": str(e)
            })
        
        return None
    
    def organize_file(self, filename: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Organize a file based on its metadata."""
        if not self.file_needs_processing(filename, 'organize'):
            logger.info(f"Skipping organization for already processed: {filename}")
            return self.state['organized_files'].get(filename)
        
        try:
            # Get source path - either from local directory or downloads
            if self.use_local_files:
                source_path = Path(self.state['downloaded_files'][filename]['path'])
            else:
                source_path = self.downloads_dir / filename
                
            if not source_path.exists():
                logger.error(f"Source file not found: {filename}")
                return None
            
            # Convert our metadata to DocumentMetadata format expected by organization engine
            from src.metadata_extraction.extractor import DocumentMetadata, DateInfo, Entity
            
            # Extract entities from AI results
            entities = []
            if 'key_entities' in metadata:
                for entity_type, entity_list in metadata.get('key_entities', {}).items():
                    for entity_name in entity_list:
                        entities.append(Entity(name=entity_name, type=entity_type))
            
            # Create DateInfo
            date_info = DateInfo(
                document_date=metadata.get('dates', {}).get('document_date'),
                mentioned_dates=metadata.get('date_references', [])
            )
            
            # Create DocumentMetadata
            doc_metadata = DocumentMetadata(
                document_type=metadata.get('document_type', 'unknown'),
                categories=metadata.get('categories', []),
                dates=date_info,
                entities=entities,
                topics=metadata.get('topics', []),
                tags=metadata.get('classification_tags', []),
                summary=metadata.get('summary', ''),
                suggested_folder=metadata.get('suggested_folder_path', ''),
                confidence_score=metadata.get('confidence_score', 0.5),
                source_file=filename,
                processing_timestamp=datetime.now().isoformat()
            )
            
            # Determine target location using organization engine
            target_path, rule_name = self.organization_engine.determine_target_location(doc_metadata)
            
            # If target_path is just a directory, append the filename
            if not target_path.endswith(filename):
                target_path = Path(target_path) / filename
            
            # Create full target path
            full_target = self.organized_dir / target_path
            full_target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to organized location
            import shutil
            shutil.copy2(source_path, full_target)
            
            # Update state
            self.state['organized_files'][filename] = str(full_target)
            self.save_state()
            
            logger.info(f"Organized {filename} -> {target_path} (rule: {rule_name})")
            return str(full_target)
            
        except Exception as e:
            logger.error(f"Error organizing {filename}: {e}")
            self.errors.append({
                "stage": "organize",
                "file": filename,
                "error": str(e)
            })
        
        return None
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        report_data = {
            "processing_summary": {
                "total_files": len(self.state['downloaded_files']),
                "ocr_processed": len(self.state['ocr_processed']),
                "ai_processed": len(self.state['ai_processed']),
                "organized": len(self.state['organized_files']),
                "errors": len(self.errors)
            },
            "errors": self.errors,
            "file_details": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # Compile file details
        for filename in self.state['downloaded_files']:
            file_detail = {
                "filename": filename,
                "download": self.state['downloaded_files'].get(filename, {}),
                "ocr": self.state['ocr_processed'].get(filename),
                "ai": self.state['ai_processed'].get(filename),
                "organized": self.state['organized_files'].get(filename)
            }
            report_data['file_details'].append(file_detail)
        
        # Save report
        report_file = self.reports_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create summary report
        summary_file = self.reports_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("Hansman Syracuse Collection Processing Report\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Processing Summary:\n")
            f.write(f"- Total files: {report_data['processing_summary']['total_files']}\n")
            f.write(f"- OCR processed: {report_data['processing_summary']['ocr_processed']}\n")
            f.write(f"- AI processed: {report_data['processing_summary']['ai_processed']}\n")
            f.write(f"- Organized: {report_data['processing_summary']['organized']}\n")
            f.write(f"- Errors: {report_data['processing_summary']['errors']}\n\n")
            
            if self.errors:
                f.write("Errors:\n")
                for error in self.errors[:10]:  # Show first 10 errors
                    f.write(f"- {error.get('stage')}: {error.get('file', 'N/A')} - {error.get('error')}\n")
                if len(self.errors) > 10:
                    f.write(f"... and {len(self.errors) - 10} more errors\n")
            
            f.write(f"\nFull report: {report_file}")
        
        logger.info(f"Reports generated: {report_file} and {summary_file}")
        return str(report_file), str(summary_file)
    
    def run_full_pipeline(self):
        """Run the complete processing pipeline."""
        start_time = time.time()
        
        logger.info("Starting Hansman collection processing...")
        print("Hansman Syracuse Collection Processing")
        print("=" * 40)
        
        if self.use_local_files:
            # Use local files
            print("Using local files mode")
            files = self.list_local_files()
            if not files:
                logger.error("No files found in local directory. Exiting.")
                return False
            
            # Apply limit if specified
            if self.download_limit:
                files = files[:self.download_limit]
                logger.info(f"Limited to {self.download_limit} files")
            
            # Create file metadata for local files
            for file_path in files:
                filename = file_path.name
                if filename not in self.state['downloaded_files']:
                    self.state['downloaded_files'][filename] = {
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "downloaded_at": datetime.now().isoformat()
                    }
            self.save_state()
            print(f"Using {len(files)} local files")
        else:
            # 1. Connect to Dropbox
            if not self.connect_to_dropbox():
                logger.error("Failed to connect to Dropbox. Exiting.")
                return False
            
            # 2. List files
            files = self.list_files()
            if not files:
                logger.error("No files found. Exiting.")
                return False
            
            # 3. Download files
            print(f"\nStep 1: Downloading files...")
            download_stats = self.download_files(files)
            print(f"Downloaded: {download_stats['downloaded']}, Skipped: {download_stats['skipped']}, Errors: {download_stats['errors']}")
        
        # 4. Initialize processors
        if not self.initialize_processors():
            logger.error("Failed to initialize processors. Exiting.")
            return False
        
        # 5. Process OCR
        print(f"\nStep 2: Processing OCR...")
        ocr_results = {}
        
        # Apply limit to processing if specified
        files_to_process = list(self.state['downloaded_files'].keys())
        if self.download_limit:
            files_to_process = files_to_process[:self.download_limit]
        
        ocr_processed = 0
        ocr_skipped = 0
        
        for filename in tqdm(files_to_process, desc="OCR Processing"):
            if not self.file_needs_processing(filename, 'ocr'):
                ocr_skipped += 1
                # Load existing OCR result
                ocr_path = self.state['ocr_processed'].get(filename)
                if ocr_path and Path(ocr_path).exists():
                    with open(ocr_path, 'r') as f:
                        ocr_results[filename] = json.load(f)
            else:
                result = self.process_ocr(filename)
                if result:
                    ocr_results[filename] = result
                    ocr_processed += 1
        
        print(f"OCR Processed: {ocr_processed}, Skipped: {ocr_skipped}")
        
        # 6. Process AI summaries
        print(f"\nStep 3: Processing AI summaries...")
        ai_results = {}
        ai_processed = 0
        ai_skipped = 0
        
        for filename, ocr_result in tqdm(ocr_results.items(), desc="AI Processing"):
            if not self.file_needs_processing(filename, 'ai'):
                ai_skipped += 1
                # Load existing AI result
                ai_path = self.state['ai_processed'].get(filename)
                if ai_path and Path(ai_path).exists():
                    with open(ai_path, 'r') as f:
                        ai_results[filename] = json.load(f)
            else:
                if ocr_result and 'text' in ocr_result:
                    result = self.process_ai_summary(filename, ocr_result['text'])
                    if result:
                        ai_results[filename] = result
                        ai_processed += 1
        
        print(f"AI Processed: {ai_processed}, Skipped: {ai_skipped}")
        
        # 7. Organize files
        print(f"\nStep 4: Organizing files...")
        organized_count = 0
        organize_skipped = 0
        
        for filename in tqdm(files_to_process, desc="Organizing"):
            if not self.file_needs_processing(filename, 'organize'):
                organize_skipped += 1
            else:
                # Get AI results if available
                ai_result = ai_results.get(filename, {})
                
                # If we have AI results, use them directly
                if ai_result:
                    metadata = ai_result
                else:
                    # Otherwise, create minimal metadata
                    metadata = {
                        'filename': filename,
                        'document_type': 'unknown',
                        'categories': [],
                        'key_entities': {},
                        'date_references': [],
                        'classification_tags': [],
                        'summary': '',
                        'suggested_folder_path': 'Hansman_Syracuse/Uncategorized',
                        'confidence_score': 0.5
                    }
                
                # Add OCR confidence if available
                ocr_result = ocr_results.get(filename, {})
                if ocr_result:
                    metadata['ocr_confidence'] = ocr_result.get('confidence', 0)
                
                # Extract keywords from filename for classification
                filename_lower = filename.lower()
                if 'syracuse' in filename_lower:
                    metadata.setdefault('classification_tags', []).append('syracuse')
                if 'list' in filename_lower:
                    metadata.setdefault('classification_tags', []).append('list')
                    metadata.setdefault('categories', []).append('list')
                if any(year in filename_lower for year in ['1950', '1951', '1952', '1953', '1954', '1955']):
                    metadata.setdefault('classification_tags', []).append('dated')
                
                result = self.organize_file(filename, metadata)
                if result:
                    organized_count += 1
        
        print(f"Organized: {organized_count}, Skipped: {organize_skipped}")
        
        # 8. Generate final report
        print(f"\nGenerating final report...")
        report_path, summary_path = self.generate_final_report()
        
        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\nProcessing Complete!")
        print(f"Total time: {hours}h {minutes}m {seconds}s")
        print(f"Report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        
        return True


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Process Hansman Syracuse photo collection')
    parser.add_argument('--output-dir', type=str, default='./hansman_results',
                        help='Output directory for results (default: ./hansman_results)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to download (default: no limit)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download of all files')
    parser.add_argument('--force-ocr', action='store_true',
                        help='Force re-run OCR on all files')
    parser.add_argument('--force-ai', action='store_true',
                        help='Force re-run AI summarization on all files')
    parser.add_argument('--force-organize', action='store_true',
                        help='Force re-organize all files')
    parser.add_argument('--force-all', action='store_true',
                        help='Force re-run all processing steps')
    parser.add_argument('--use-local', action='store_true',
                        help='Use local files instead of Dropbox')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='Local directory containing Hansman files (required when --use-local is set)')
    
    args = parser.parse_args()
    
    # Handle force-all flag
    if args.force_all:
        args.force_download = True
        args.force_ocr = True
        args.force_ai = True
        args.force_organize = True
    
    # Validate local directory when using local files
    if args.use_local and not args.local_dir:
        parser.error("--local-dir is required when using --use-local")
    
    if args.use_local:
        local_path = Path(args.local_dir)
        if not local_path.exists() or not local_path.is_dir():
            parser.error(f"Local directory '{args.local_dir}' does not exist or is not a directory")
    
    # Create processor
    processor = HansmanUnifiedProcessor(
        output_dir=args.output_dir,
        download_limit=args.limit,
        force_redownload=args.force_download,
        force_reocr=args.force_ocr,
        force_reai=args.force_ai,
        force_reorganize=args.force_organize,
        use_local_files=args.use_local,
        local_dir=args.local_dir
    )
    
    # Run pipeline
    success = processor.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()