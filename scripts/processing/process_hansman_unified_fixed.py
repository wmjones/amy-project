#!/usr/bin/env python3
"""Unified processing script for Hansman Syracuse photo collection with step checking and force options.

Enhanced with better JSON error handling for resuming from existing results.
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
from src.claude_integration.client_fixed import ClaudeClient
from src.utils.report_generator import ReportGenerator
from src.organization_logic.engine import OrganizationEngine
from src.file_access.processor import FileProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hansman_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class HansmanUnifiedProcessor:
    """Unified processing pipeline for Hansman collection with resume capability."""

    def __init__(
        self,
        output_dir: str = "./hansman_results",
        download_limit: Optional[int] = None,
        force_redownload: bool = False,
        force_reocr: bool = False,
        force_reai: bool = False,
        force_reorganize: bool = False,
        use_local_files: bool = False,
        local_dir: Optional[str] = None,
    ):
        """Initialize the processing pipeline with control options.

        Args:
            output_dir: Base output directory
            download_limit: Limit number of files to download
            force_redownload: Force redownload even if files exist
            force_reocr: Force reprocess OCR even if results exist
            force_reai: Force reprocess AI even if results exist
            force_reorganize: Force reorganize files even if already organized
            use_local_files: Use local files instead of downloading from Dropbox
            local_dir: Directory containing local files (if use_local_files is True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Control options
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

        for dir_path in [
            self.downloads_dir,
            self.ocr_results_dir,
            self.summaries_dir,
            self.reports_dir,
            self.organized_dir,
            self.state_dir,
        ]:
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
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading state: {e}, starting fresh")

        return {
            "downloaded_files": {},  # filename -> metadata
            "ocr_processed": {},  # filename -> ocr_result_path
            "ai_processed": {},  # filename -> summary_path
            "organized_files": {},  # filename -> target_path
            "errors": [],
            "last_run": None,
        }

    def save_state(self):
        """Save processing state to file."""
        self.state["last_run"] = datetime.now().isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def file_needs_processing(self, filename: str, stage: str) -> bool:
        """Check if a file needs processing for a given stage.

        Args:
            filename: File to check
            stage: Processing stage (download, ocr, ai, organize)

        Returns:
            True if file needs processing
        """
        if stage == "download":
            if self.force_redownload:
                return True
            return filename not in self.state["downloaded_files"]

        elif stage == "ocr":
            if self.force_reocr:
                return True
            return filename not in self.state["ocr_processed"]

        elif stage == "ai":
            if self.force_reai:
                return True
            return filename not in self.state["ai_processed"]

        elif stage == "organize":
            if self.force_reorganize:
                return True
            return filename not in self.state["organized_files"]

        return True

    def scan_dropbox_folder(
        self, folder_path: str = "/Hansman Syracuse photo docs July 2015"
    ) -> List[Dict[str, Any]]:
        """Scan Dropbox folder for Hansman Syracuse photo documents."""
        logger.info(f"Scanning Dropbox folder: {folder_path}")

        try:
            result = self.dbx.files_list_folder(folder_path, recursive=True)
            files = []

            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        # Filter for supported image files
                        if entry.name.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".gif")
                        ):
                            files.append(
                                {
                                    "name": entry.name,
                                    "path": entry.path_display,
                                    "size": entry.size,
                                    "modified": str(entry.server_modified),
                                    "id": entry.id,
                                }
                            )

                # Get more results if available
                if result.has_more:
                    result = self.dbx.files_list_folder_continue(result.cursor)
                else:
                    break

            logger.info(f"Found {len(files)} Hansman Syracuse files")
            return files

        except Exception as e:
            logger.error(f"Error scanning Dropbox: {e}")
            return []

    def scan_local_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan local directory for Hansman Syracuse photo documents."""
        logger.info(f"Scanning local directory: {directory}")

        files = []
        supported_extensions = {".jpg", ".jpeg", ".png", ".gif"}

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(),
                        "local": True,
                    }
                )

        logger.info(f"Found {len(files)} supported files in local directory")
        return files

    def download_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Download a file from Dropbox if needed."""
        filename = file_info["name"]

        if not self.file_needs_processing(filename, "download"):
            logger.info(f"Skipping already downloaded: {filename}")
            return self.state["downloaded_files"][filename]["path"]

        try:
            local_path = self.downloads_dir / filename

            logger.info(f"Downloading: {filename}")
            self.dbx.files_download_to_file(str(local_path), file_info["path"])

            # Update state
            self.state["downloaded_files"][filename] = {
                "path": str(local_path),
                "size": file_info["size"],
                "modified": file_info["modified"],
                "downloaded_at": datetime.now().isoformat(),
            }
            self.save_state()

            return str(local_path)

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            self.errors.append({"stage": "download", "file": filename, "error": str(e)})
            return None

    def copy_local_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Copy a local file to the downloads directory if needed."""
        filename = file_info["name"]

        if not self.file_needs_processing(filename, "download"):
            logger.info(f"Skipping already copied: {filename}")
            return self.state["downloaded_files"][filename]["path"]

        try:
            source_path = Path(file_info["path"])
            local_path = self.downloads_dir / filename

            logger.info(f"Copying: {filename}")
            import shutil

            shutil.copy2(source_path, local_path)

            # Update state
            self.state["downloaded_files"][filename] = {
                "path": str(local_path),
                "size": file_info["size"],
                "modified": file_info["modified"],
                "downloaded_at": datetime.now().isoformat(),
            }
            self.save_state()

            return str(local_path)

        except Exception as e:
            logger.error(f"Error copying {filename}: {e}")
            self.errors.append({"stage": "copy", "file": filename, "error": str(e)})
            return None

    def add_hansman_organization_rules(self):
        """Add custom organization rules for Hansman collection."""
        # Rule for Syracuse lists
        self.organization_engine.add_rule(
            {
                "name": "Syracuse Lists",
                "conditions": {"keywords": ["syracuse list", "syracuse directory"]},
                "path_template": "Syracuse_Lists/{filename}",
                "priority": 100,
            }
        )

        # Rule for dated photos
        self.organization_engine.add_rule(
            {
                "name": "Dated Photos",
                "conditions": {"file_pattern": r"(\d{4})_", "document_type": "photo"},
                "path_template": "Photos/{dates.year}/{filename}",
                "priority": 90,
            }
        )

        # Rule for sequential photos
        self.organization_engine.add_rule(
            {
                "name": "Sequential Photos",
                "conditions": {"file_pattern": r"^100_\d+"},
                "path_template": "Sequential_Photos/{filename}",
                "priority": 85,
            }
        )

        # Rule for correspondence
        self.organization_engine.add_rule(
            {
                "name": "Correspondence",
                "conditions": {
                    "keywords": ["arnold", "knowles", "letter", "correspondence"]
                },
                "path_template": "Correspondence/{filename}",
                "priority": 80,
            }
        )

    def process_ocr(self, filename: str) -> Optional[Dict[str, Any]]:
        """Process a single file through OCR using enhanced preprocessing and OCR."""
        if not self.file_needs_processing(filename, "ocr"):
            logger.info(f"Skipping OCR for already processed: {filename}")
            return self.state["ocr_processed"].get(filename)

        try:
            # Get file path - either from local directory or downloads
            if self.use_local_files:
                local_path = Path(self.state["downloaded_files"][filename]["path"])
            else:
                local_path = self.downloads_dir / filename

            if not local_path.exists():
                logger.error(f"File not found for OCR: {filename}")
                return None

            logger.info(f"Processing OCR for: {filename}")

            # First, run preprocessing pipeline to get enhanced image variants
            preprocessing_result = self.preprocessing_pipeline.process_document(
                str(local_path), str(self.ocr_results_dir / local_path.stem)
            )

            # Get the enhanced image variants
            enhanced_images_paths = preprocessing_result.get("enhanced_images", {})
            metadata = preprocessing_result.get("metadata", None)

            # Load the enhanced images as numpy arrays
            enhanced_images = {}
            image_enhancer = self.preprocessing_pipeline.enhancer

            # Get the enhanced images directly from the enhancer
            enhanced_images = image_enhancer.enhance_image(str(local_path))

            # Process variants through enhanced OCR
            ocr_result = self.ocr_processor.process_variants(
                enhanced_images, str(local_path)
            )

            if ocr_result:
                # Save OCR text
                ocr_file = self.ocr_results_dir / f"{local_path.stem}_ocr.txt"
                with open(ocr_file, "w", encoding="utf-8") as f:
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
                            "suggested_action": region.suggested_action,
                        }
                        for region in ocr_result.problematic_regions
                    ],
                    "variant_results": ocr_result.variant_results,
                    "preprocessing_metadata": metadata.to_dict() if metadata else None,
                    "processed_at": datetime.now().isoformat(),
                }

                # Custom JSON encoder for numpy types
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if hasattr(obj, "item"):  # numpy scalar
                            return obj.item()
                        if hasattr(obj, "tolist"):  # numpy array
                            return obj.tolist()
                        return super().default(obj)

                with open(ocr_json_file, "w") as f:
                    json.dump(result_data, f, indent=2, cls=NumpyEncoder)

                # Export structured OCR output
                ocr_export_file = (
                    self.ocr_results_dir / f"{local_path.stem}_ocr_structured.json"
                )
                self.ocr_processor.export_structured_output(
                    ocr_result, str(ocr_export_file), format="json"
                )

                # Update state
                self.state["ocr_processed"][filename] = str(ocr_json_file)
                self.save_state()

                return result_data

        except Exception as e:
            logger.error(f"Error in OCR processing for {filename}: {e}")
            self.errors.append({"stage": "ocr", "file": filename, "error": str(e)})

        return None

    def process_ai_summary(
        self, filename: str, ocr_text: str
    ) -> Optional[Dict[str, Any]]:
        """Process a file through AI summarization."""
        if not self.file_needs_processing(filename, "ai"):
            logger.info(f"Skipping AI for already processed: {filename}")
            return self.state["ai_processed"].get(filename)

        try:
            logger.info(f"Processing AI summary for: {filename}")

            # Use the AI summarizer's document method
            from pathlib import Path

            # Get file path - either from local directory or downloads
            if self.use_local_files:
                file_path = Path(self.state["downloaded_files"][filename]["path"])
            else:
                file_path = self.downloads_dir / filename

            # Log if image will be attached
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            if file_path.suffix.lower() in image_extensions:
                logger.info(f"📷 Including original image in AI analysis: {filename}")

            summary_result = self.ai_summarizer.summarize_document(
                ocr_text=ocr_text,
                file_path=file_path,
                additional_context={
                    "collection": "Hansman Syracuse photo docs July 2015",
                    "processing_date": datetime.now().isoformat(),
                },
            )

            if summary_result:
                # Save AI summary as structured data
                summary_file = (
                    self.summaries_dir / f"{Path(filename).stem}_summary.json"
                )

                # Convert DocumentSummary dataclass to dict
                from dataclasses import asdict

                summary_data = asdict(summary_result)

                # Convert datetime objects to strings
                if "created_at" in summary_data and hasattr(
                    summary_data["created_at"], "isoformat"
                ):
                    summary_data["created_at"] = summary_data["created_at"].isoformat()

                summary_data["processed_at"] = datetime.now().isoformat()

                with open(summary_file, "w") as f:
                    json.dump(
                        summary_data, f, indent=2, default=str
                    )  # Use default=str for any remaining datetime objects

                # Update state
                self.state["ai_processed"][filename] = str(summary_file)
                self.save_state()

                return summary_data

        except Exception as e:
            logger.error(f"Error in AI processing for {filename}: {e}")
            self.errors.append({"stage": "ai", "file": filename, "error": str(e)})

        return None

    def organize_file(self, filename: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Organize a file based on its metadata."""
        if not self.file_needs_processing(filename, "organize"):
            logger.info(f"Skipping organization for already processed: {filename}")
            return self.state["organized_files"].get(filename)

        try:
            # Get source path - either from local directory or downloads
            if self.use_local_files:
                source_path = Path(self.state["downloaded_files"][filename]["path"])
            else:
                source_path = self.downloads_dir / filename

            if not source_path.exists():
                logger.error(f"Source file not found: {filename}")
                return None

            # Convert our metadata to DocumentMetadata format expected by organization engine
            from src.metadata_extraction.extractor import (
                DocumentMetadata,
                DateInfo,
                Entity,
            )

            # Extract entities from AI results
            entities = []
            if "key_entities" in metadata:
                for entity_type, entity_list in metadata.get(
                    "key_entities", {}
                ).items():
                    for entity_name in entity_list:
                        entities.append(Entity(name=entity_name, type=entity_type))

            # Create DateInfo
            date_info = DateInfo(
                document_date=metadata.get("dates", {}).get("document_date"),
                mentioned_dates=metadata.get("date_references", []),
            )

            # Create DocumentMetadata
            doc_metadata = DocumentMetadata(
                document_type=metadata.get("document_type", "unknown"),
                categories=metadata.get("categories", []),
                dates=date_info,
                entities=entities,
                topics=metadata.get("topics", []),
                tags=metadata.get("classification_tags", []),
                summary=metadata.get("summary", ""),
                confidence_scores={"overall": metadata.get("confidence_score", 0.0)},
            )

            # Create OrganizationContext
            from src.organization_logic.engine import OrganizationContext

            context = OrganizationContext(
                file_path=str(source_path),
                current_path="",
                metadata=doc_metadata,
                custom_data={
                    "collection": "Hansman Syracuse July 2015",
                    "suggested_path": metadata.get("suggested_folder_path", ""),
                },
            )

            # Apply organization rules
            decisions = self.organization_engine.apply_rules(context)

            # Get the best decision
            if decisions:
                best_decision = max(decisions, key=lambda d: d.priority)
                target_path = best_decision.new_path

                # Ensure the path is within our organized directory
                if not target_path.startswith(str(self.organized_dir)):
                    target_path = self.organized_dir / target_path
                else:
                    target_path = Path(target_path)

                # Create target directory
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file to new location
                import shutil

                shutil.copy2(source_path, target_path)

                logger.info(f"Organized {filename} to {target_path}")

                # Update state
                self.state["organized_files"][filename] = str(target_path)
                self.save_state()

                return str(target_path)
            else:
                # Default fallback
                target_path = self.organized_dir / "Uncategorized" / filename
                target_path.parent.mkdir(parents=True, exist_ok=True)

                import shutil

                shutil.copy2(source_path, target_path)

                # Update state
                self.state["organized_files"][filename] = str(target_path)
                self.save_state()

                return str(target_path)

        except Exception as e:
            logger.error(f"Error organizing {filename}: {e}")
            self.errors.append({"stage": "organize", "file": filename, "error": str(e)})

        return None

    def initialize_processors(self):
        """Initialize all processing components."""
        logger.info("Initializing processors...")

        # Initialize preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline()

        # Initialize enhanced OCR processor
        self.ocr_processor = EnhancedOCRProcessor(
            language="eng",
            confidence_threshold=60.0,
            parallel_processing=True,
            max_workers=4,
            cache_enabled=True,
            cache_dir=str(self.state_dir / "ocr_cache"),
        )

        # Initialize Claude client
        claude_client = ClaudeClient()

        # Initialize AI summarizer with fixed version
        self.ai_summarizer = AISummarizer(claude_client=claude_client)

        # Initialize organization engine
        self.organization_engine = OrganizationEngine()
        self.add_hansman_organization_rules()

        logger.info("All processors initialized successfully")

    def run_full_pipeline(self) -> bool:
        """Run the complete processing pipeline for all files (without organization)."""
        print("Hansman Syracuse Collection Processing (OCR + AI Only)")
        print("====================================================")

        # 1. Initialize processors
        self.initialize_processors()

        # 2. Get files to process
        if self.use_local_files:
            print(f"Using local files mode")
            if not self.local_dir or not self.local_dir.exists():
                print(f"Error: Local directory not found: {self.local_dir}")
                return False

            files = self.scan_local_directory(self.local_dir)
        else:
            print("Connecting to Dropbox...")
            try:
                self.dbx = dropbox.Dropbox(os.getenv("DROPBOX_ACCESS_TOKEN"))
                self.dbx.users_get_current_account()
            except Exception as e:
                print(f"Failed to connect to Dropbox: {e}")
                return False

            files = self.scan_dropbox_folder()

        if not files:
            print("No files found to process")
            return False

        # Apply limit if specified
        if self.download_limit:
            files = files[: self.download_limit]
            print(f"Limited to {self.download_limit} files")

        print(
            f"Using {len(files)} {'local' if self.use_local_files else 'Dropbox'} files"
        )

        # 3. Download/Copy files
        if self.use_local_files:
            print(f"\nStep 1: Copying local files...")
            download_results = {}
            for file_info in files:
                result = self.copy_local_file(file_info)
                if result:
                    download_results[file_info["name"]] = result
        else:
            print(f"\nStep 1: Downloading files from Dropbox...")
            download_results = {}
            with tqdm(total=len(files), desc="Downloading") as pbar:
                for file_info in files:
                    result = self.download_file(file_info)
                    if result:
                        download_results[file_info["name"]] = result
                    pbar.update(1)

        print(
            f"{'Copied' if self.use_local_files else 'Downloaded'}: {len(download_results)} files"
        )

        # 4. Load existing results if resuming
        # First load OCR results from state with better error handling
        ocr_results = {}
        print(f"\nStep 2: Processing OCR...")
        for filename in download_results.keys():
            if not self.file_needs_processing(filename, "ocr"):
                # Load existing OCR result with error handling
                ocr_path = self.state["ocr_processed"].get(filename)
                if ocr_path and Path(ocr_path).exists():
                    try:
                        with open(ocr_path, "r") as f:
                            ocr_results[filename] = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to load OCR JSON for {filename}: {e}. Will reprocess."
                        )
                        # Force reprocess this file
                        result = self.process_ocr(filename)
                        if result:
                            ocr_results[filename] = result
                    except Exception as e:
                        logger.warning(
                            f"Error loading OCR for {filename}: {e}. Will reprocess."
                        )
                        result = self.process_ocr(filename)
                        if result:
                            ocr_results[filename] = result
                else:
                    logger.warning(
                        f"OCR path not found for {filename}. Will reprocess."
                    )
                    result = self.process_ocr(filename)
                    if result:
                        ocr_results[filename] = result
            else:
                result = self.process_ocr(filename)
                if result:
                    ocr_results[filename] = result

        print(f"OCR Processed: {len(ocr_results)} files")

        # 5. Process AI summaries
        print(f"\nStep 3: Processing AI summaries...")
        ai_results = {}
        ai_processed = 0
        ai_skipped = 0

        with tqdm(total=len(ocr_results), desc="AI Processing") as pbar:
            for filename, ocr_data in ocr_results.items():
                if not self.file_needs_processing(filename, "ai"):
                    ai_skipped += 1
                    # Load existing AI result with error handling
                    ai_path = self.state["ai_processed"].get(filename)
                    if ai_path and Path(ai_path).exists():
                        try:
                            with open(ai_path, "r") as f:
                                ai_results[filename] = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to load AI JSON for {filename}: {e}. Will reprocess."
                            )
                            # Get OCR text
                            ocr_text = (
                                ocr_data.get("text", "")
                                if isinstance(ocr_data, dict)
                                else str(ocr_data)
                            )
                            result = self.process_ai_summary(filename, ocr_text)
                            if result:
                                ai_results[filename] = result
                                ai_processed += 1
                        except Exception as e:
                            logger.warning(
                                f"Error loading AI summary for {filename}: {e}. Will reprocess."
                            )
                            ocr_text = (
                                ocr_data.get("text", "")
                                if isinstance(ocr_data, dict)
                                else str(ocr_data)
                            )
                            result = self.process_ai_summary(filename, ocr_text)
                            if result:
                                ai_results[filename] = result
                                ai_processed += 1
                else:
                    # Get OCR text safely
                    ocr_text = (
                        ocr_data.get("text", "")
                        if isinstance(ocr_data, dict)
                        else str(ocr_data)
                    )
                    result = self.process_ai_summary(filename, ocr_text)
                    if result:
                        ai_results[filename] = result
                        ai_processed += 1
                pbar.update(1)

        print(f"AI Processed: {ai_processed}, Skipped: {ai_skipped}")

        # 6. Generate reports (modified to exclude organization)
        self.generate_reports()

        # 7. Print summary (modified to exclude organization)
        print("\nProcessing Summary:")
        print(f"- Total files: {len(files)}")
        print(f"- Downloaded/Copied: {len(download_results)}")
        print(f"- OCR processed: {len(ocr_results)}")
        print(f"- AI processed: {len(ai_results)}")
        print(f"- Errors: {len(self.errors)}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error['stage']}: {error['file']} - {error['error']}")

        print("\nNote: File organization step has been skipped.")
        print("All files remain in their downloaded/copied locations.")

        return True

    def generate_reports(self):
        """Generate final reports for the processing session."""
        try:
            # Create summary report
            report_data = {
                "processing_summary": {
                    "total_files": len(self.state["downloaded_files"]),
                    "ocr_processed": len(self.state["ocr_processed"]),
                    "ai_processed": len(self.state["ai_processed"]),
                    "errors": len(self.errors),
                    "note": "File organization step was skipped",
                },
                "errors": self.errors,
                "file_details": [],
            }

            # Add details for each file
            for filename in self.state["downloaded_files"]:
                file_detail = {
                    "filename": filename,
                    "download": self.state["downloaded_files"].get(filename),
                    "ocr": self.state["ocr_processed"].get(filename),
                    "ai": self.state["ai_processed"].get(filename),
                }
                report_data["file_details"].append(file_detail)

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"final_report_{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)

            # Create text summary
            summary_file = self.reports_dir / f"summary_{timestamp}.txt"
            with open(summary_file, "w") as f:
                f.write(
                    "Hansman Syracuse Collection Processing Summary (OCR + AI Only)\n"
                )
                f.write("=" * 60 + "\n\n")
                f.write(
                    f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Total Files: {len(self.state['downloaded_files'])}\n")
                f.write(f"OCR Processed: {len(self.state['ocr_processed'])}\n")
                f.write(f"AI Processed: {len(self.state['ai_processed'])}\n")
                f.write(f"Errors: {len(self.errors)}\n")
                f.write("\nNote: File organization step was skipped.\n")

                if self.errors:
                    f.write("\n\nErrors:\n")
                    for error in self.errors:
                        f.write(
                            f"- {error['stage']}: {error['file']} - {error['error']}\n"
                        )

            logger.info(f"Reports generated: {report_file} and {summary_file}")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process Hansman Syracuse photo collection with OCR and AI analysis (without file organization)"
    )

    # Directory options
    parser.add_argument(
        "--output-dir",
        default="./hansman_results",
        help="Output directory for results (default: ./hansman_results)",
    )

    # Processing options
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files to process"
    )

    # Force options
    parser.add_argument(
        "--force-download", action="store_true", help="Force redownload all files"
    )
    parser.add_argument(
        "--force-ocr", action="store_true", help="Force rerun OCR on all files"
    )
    parser.add_argument(
        "--force-ai", action="store_true", help="Force rerun AI analysis on all files"
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force rerun everything (excluding organization)",
    )

    # Local file options
    parser.add_argument(
        "--use-local", action="store_true", help="Use local files instead of Dropbox"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory containing files (required with --use-local)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.use_local and not args.local_dir:
        parser.error("--local-dir is required when using --use-local")

    # Apply force-all
    if args.force_all:
        args.force_download = True
        args.force_ocr = True
        args.force_ai = True

    logger.info("Starting Hansman collection processing (OCR + AI only)...")

    processor = HansmanUnifiedProcessor(
        output_dir=args.output_dir,
        download_limit=args.limit,
        force_redownload=args.force_download,
        force_reocr=args.force_ocr,
        force_reai=args.force_ai,
        force_reorganize=False,  # Always False now
        use_local_files=args.use_local,
        local_dir=args.local_dir,
    )

    success = processor.run_full_pipeline()

    if success:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing encountered errors")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
