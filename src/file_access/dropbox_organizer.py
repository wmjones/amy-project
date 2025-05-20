"""
Dropbox-specific file organization functionality.
Integrates Dropbox API with the file organization system.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .dropbox_accessor import DropboxAccessor, DropboxFile
from .file_system_accessor import FileSystemAccessor
from ..metadata.extractor import MetadataExtractor
from ..organization.organization_engine import OrganizationEngine
from ..utils.file_processor import FileProcessor
from ..utils.report_generator import ReportGenerator
from ..utils.progress_tracker import ProgressTracker
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DropboxOrganizer:
    """Organize files within Dropbox using the file organization system."""

    def __init__(
        self,
        config: ConfigManager,
        dropbox_accessor: DropboxAccessor,
        metadata_extractor: MetadataExtractor,
        organization_engine: OrganizationEngine,
        file_processor: FileProcessor,
        report_generator: ReportGenerator,
    ):
        """
        Initialize Dropbox organizer.

        Args:
            config: Configuration manager
            dropbox_accessor: Dropbox API accessor
            metadata_extractor: Metadata extraction service
            organization_engine: Organization logic engine
            file_processor: File processing service
            report_generator: Report generation service
        """
        self.config = config
        self.dropbox = dropbox_accessor
        self.metadata_extractor = metadata_extractor
        self.organization_engine = organization_engine
        self.file_processor = file_processor
        self.report_generator = report_generator

        # Configuration
        self.source_folder = config.get(
            "dropbox.source_folder", "/file-organizer-uploads"
        )
        self.organized_folder = config.get("dropbox.organized_folder", "/Organized")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dropbox_org_"))
        self.batch_size = config.get("dropbox.download_batch_size", 10)
        self.cleanup_after_org = config.get("dropbox.cleanup_after_organization", False)

    def organize_dropbox_files(
        self,
        source_folder: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> Dict[str, Any]:
        """
        Organize files in Dropbox.

        Args:
            source_folder: Dropbox folder to organize (uses config default if None)
            file_types: List of file extensions to process
            progress_tracker: Progress tracking instance

        Returns:
            Dictionary with organization results
        """
        source_folder = source_folder or self.source_folder
        file_types = file_types or self.config.get("processing.file_types")

        logger.info(f"Starting Dropbox organization from {source_folder}")

        try:
            # List files to process
            dropbox_files = self.dropbox.list_files(
                source_folder, recursive=True, file_types=file_types
            )

            if not dropbox_files:
                logger.info("No files found to organize")
                return {"processed": 0, "successful": 0, "failed": 0, "skipped": 0}

            logger.info(f"Found {len(dropbox_files)} files to organize")

            # Initialize progress tracking
            if progress_tracker:
                progress_tracker.set_total(len(dropbox_files))

            # Process files in batches
            results = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "movements": [],
            }

            for i in range(0, len(dropbox_files), self.batch_size):
                batch = dropbox_files[i : i + self.batch_size]
                batch_results = self._process_batch(batch, progress_tracker)

                # Aggregate results
                results["processed"] += batch_results["processed"]
                results["successful"] += batch_results["successful"]
                results["failed"] += batch_results["failed"]
                results["skipped"] += batch_results["skipped"]
                results["movements"].extend(batch_results["movements"])

            logger.info(
                f"Organization complete. Processed: {results['processed']}, "
                f"Successful: {results['successful']}, Failed: {results['failed']}"
            )

            return results

        finally:
            # Cleanup temporary directory
            if self.temp_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up temp directory: {e}")

    def _process_batch(
        self, batch: List[DropboxFile], progress_tracker: Optional[ProgressTracker]
    ) -> Dict[str, Any]:
        """Process a batch of files."""
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "movements": [],
        }

        # Download files to temp directory
        download_results = self._download_batch(batch)

        for dropbox_file, local_path in download_results.items():
            if local_path is None:
                results["failed"] += 1
                if progress_tracker:
                    progress_tracker.update_file(
                        dropbox_file.path, "failed", error="Download failed"
                    )
                continue

            try:
                # Process the file
                result = self._process_single_file(dropbox_file, local_path)

                if result["success"]:
                    results["successful"] += 1
                    results["movements"].append(result["movement"])
                else:
                    results["failed"] += 1

                results["processed"] += 1

                if progress_tracker:
                    progress_tracker.update_file(
                        dropbox_file.path,
                        "completed" if result["success"] else "failed",
                        error=result.get("error"),
                    )

            except Exception as e:
                logger.error(f"Error processing {dropbox_file.path}: {e}")
                results["failed"] += 1
                results["processed"] += 1

                if progress_tracker:
                    progress_tracker.update_file(
                        dropbox_file.path, "failed", error=str(e)
                    )

            finally:
                # Clean up local file
                try:
                    if local_path.exists():
                        local_path.unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up {local_path}: {e}")

        return results

    def _download_batch(
        self, batch: List[DropboxFile]
    ) -> Dict[DropboxFile, Optional[Path]]:
        """Download a batch of files from Dropbox."""
        results = {}

        for dropbox_file in batch:
            try:
                local_filename = f"{dropbox_file.id}_{dropbox_file.name}"
                local_path = self.temp_dir / local_filename

                self.dropbox.download_file(
                    dropbox_file.path, local_path, show_progress=False
                )

                results[dropbox_file] = local_path

            except Exception as e:
                logger.error(f"Error downloading {dropbox_file.path}: {e}")
                results[dropbox_file] = None

        return results

    def _process_single_file(
        self, dropbox_file: DropboxFile, local_path: Path
    ) -> Dict[str, Any]:
        """Process a single file."""
        try:
            # Extract metadata
            processed_file = self.file_processor.process_file(local_path)
            if not processed_file:
                return {"success": False, "error": "Failed to process file"}

            metadata = self.metadata_extractor.extract_from_processed_file(
                processed_file
            )

            # Add Dropbox-specific metadata
            metadata["dropbox"] = {
                "id": dropbox_file.id,
                "original_path": dropbox_file.path,
                "modified": dropbox_file.modified.isoformat(),
                "size": dropbox_file.size,
            }

            # Determine organization path
            rule = self.organization_engine.find_matching_rule(metadata)
            if not rule:
                return {"success": False, "error": "No matching organization rule"}

            # Generate organized path
            organized_path = self.organization_engine.generate_organized_path(
                Path(dropbox_file.path).name, metadata, rule
            )

            # Construct full Dropbox path
            dropbox_organized_path = f"{self.organized_folder}/{organized_path}"

            # Create necessary folders
            self._ensure_dropbox_path(Path(dropbox_organized_path).parent)

            # Move or copy file in Dropbox
            if self.config.get("organization.mode") == "move":
                success = self.dropbox.move_file(
                    dropbox_file.path, dropbox_organized_path, autorename=True
                )
            else:
                success = self.dropbox.copy_file(
                    dropbox_file.path, dropbox_organized_path, autorename=True
                )

            if success:
                # Record movement
                movement = {
                    "source": dropbox_file.path,
                    "destination": dropbox_organized_path,
                    "metadata": metadata,
                    "rule": rule["name"],
                }

                self.report_generator.record_file_movement(
                    Path(dropbox_file.path),
                    Path(dropbox_organized_path),
                    metadata,
                    rule["name"],
                )

                # Clean up original if configured
                if (
                    self.cleanup_after_org
                    and self.config.get("organization.mode") == "copy"
                ):
                    try:
                        self.dropbox.delete_file(dropbox_file.path)
                    except Exception as e:
                        logger.warning(f"Failed to delete original file: {e}")

                return {"success": True, "movement": movement}
            else:
                return {
                    "success": False,
                    "error": "Failed to move/copy file in Dropbox",
                }

        except Exception as e:
            logger.error(f"Error processing {dropbox_file.path}: {e}")
            return {"success": False, "error": str(e)}

    def _ensure_dropbox_path(self, path: Path):
        """Ensure all folders in the path exist in Dropbox."""
        # Build path components
        parts = path.parts
        current_path = ""

        for part in parts:
            if not part:  # Skip empty parts
                continue

            current_path = f"{current_path}/{part}" if current_path else f"/{part}"

            try:
                self.dropbox.create_folder(current_path)
            except Exception as e:
                # Folder might already exist, which is fine
                logger.debug(f"Folder creation for {current_path}: {e}")

    def scan_and_preview(
        self,
        source_folder: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Scan Dropbox and preview organization without making changes.

        Args:
            source_folder: Folder to scan
            file_types: File types to include

        Returns:
            Preview of organization actions
        """
        source_folder = source_folder or self.source_folder
        file_types = file_types or self.config.get("processing.file_types")

        dropbox_files = self.dropbox.list_files(
            source_folder, recursive=True, file_types=file_types
        )

        preview = {
            "total_files": len(dropbox_files),
            "by_type": {},
            "by_size": {"< 1MB": 0, "1-10MB": 0, "10-100MB": 0, "> 100MB": 0},
            "sample_movements": [],
        }

        # Analyze files
        for file in dropbox_files:
            # Count by type
            ext = file.extension or "no_extension"
            preview["by_type"][ext] = preview["by_type"].get(ext, 0) + 1

            # Count by size
            if file.size < 1024 * 1024:
                preview["by_size"]["< 1MB"] += 1
            elif file.size < 10 * 1024 * 1024:
                preview["by_size"]["1-10MB"] += 1
            elif file.size < 100 * 1024 * 1024:
                preview["by_size"]["10-100MB"] += 1
            else:
                preview["by_size"]["> 100MB"] += 1

        # Generate sample movements (first 5 files)
        for file in dropbox_files[:5]:
            try:
                # Download small sample for preview
                temp_path = self.temp_dir / f"preview_{file.id}"
                self.dropbox.download_file(file.path, temp_path, show_progress=False)

                # Process for preview
                processed = self.file_processor.process_file(temp_path)
                if processed:
                    metadata = self.metadata_extractor.extract_from_processed_file(
                        processed
                    )
                    rule = self.organization_engine.find_matching_rule(metadata)

                    if rule:
                        organized_path = (
                            self.organization_engine.generate_organized_path(
                                Path(file.path).name, metadata, rule
                            )
                        )

                        preview["sample_movements"].append(
                            {
                                "source": file.path,
                                "destination": f"{self.organized_folder}/{organized_path}",
                                "rule": rule["name"],
                            }
                        )

                # Clean up
                if temp_path.exists():
                    temp_path.unlink()

            except Exception as e:
                logger.warning(f"Error generating preview for {file.path}: {e}")

        return preview

    def get_organization_stats(self) -> Dict[str, Any]:
        """Get statistics about organized files in Dropbox."""
        try:
            organized_files = self.dropbox.list_files(
                self.organized_folder, recursive=True
            )

            stats = {
                "total_organized": len(organized_files),
                "by_folder": {},
                "total_size": sum(f.size for f in organized_files),
            }

            # Count files by top-level folder
            for file in organized_files:
                parts = Path(file.path).parts
                if len(parts) > 1:
                    folder = parts[1]  # First folder after organized root
                    stats["by_folder"][folder] = stats["by_folder"].get(folder, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting organization stats: {e}")
            return {}
