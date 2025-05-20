"""
Main application controller for the file organization system.
Orchestrates all components and implements the core workflow.
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# Core components
from src.utils.config_manager import ConfigManager
from src.file_access.local_accessor import FileSystemAccessor
from src.file_access.dropbox_accessor import DropboxAccessor
from src.file_access.processor import FileProcessor
from src.file_access.manipulator import FileManipulator
from src.claude_integration.client import ClaudeClient
from src.metadata_extraction.extractor import MetadataExtractor
from src.organization_logic.engine import OrganizationEngine
from src.utils.simple_batch_processor import SimpleBatchProcessor
from src.utils.error_handler import ErrorHandler
from src.utils.progress import ProgressTracker
from src.utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class FileOrganizerApp:
    """Main application controller that orchestrates file organization."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the application with configuration.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config_manager = None
        self.config = None
        self.components = {}
        self.progress_tracker = None
        self._is_initialized = False

    def initialize(self):
        """Initialize all components and load configuration."""
        if self._is_initialized:
            return

        try:
            # Load configuration
            logger.info("Loading configuration...")
            self.config_manager = ConfigManager(
                config_file=Path(self.config_file) if self.config_file else None
            )
            self.config = self.config_manager.config

            # Set up logging based on config
            self._setup_logging()

            # Initialize components
            logger.info("Initializing components...")
            self._initialize_components()

            self._is_initialized = True
            logger.info("Application initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Configure logging based on application settings."""
        log_config = self.config.get("logging", {})

        # Create logs directory if needed
        log_file = log_config.get("file", "./logs/organizer.log")
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

    def _initialize_components(self):
        """Initialize all application components."""
        # Initialize file accessor based on source type
        source_config = self.config.get("source", {})
        if source_config.get("type") == "dropbox":
            dropbox_config = self.config.get("dropbox", {})
            self.components["file_accessor"] = DropboxAccessor(
                app_key=dropbox_config["app_key"],
                app_secret=dropbox_config["app_secret"],
                refresh_token=dropbox_config.get("refresh_token"),
                access_token=dropbox_config.get("access_token"),
            )
        else:
            source_dir = source_config.get("directory", ".")
            self.components["file_accessor"] = FileSystemAccessor(source_dir)

        # Initialize file processor
        processing_config = self.config.get("processing", {})
        self.components["file_processor"] = FileProcessor(
            enable_ocr=processing_config.get("enable_ocr", True)
        )

        # Initialize Claude client
        api_config = self.config.get("api", {})
        self.components["claude_client"] = ClaudeClient(
            api_key=api_config.get("anthropic_api_key"),
            model=api_config.get("claude_model", "claude-3-opus-20240229"),
            max_tokens=api_config.get("max_tokens", 2000),
            retry_attempts=api_config.get("max_retries", 3),
            retry_delay=api_config.get("initial_backoff", 1.0),
        )

        # Initialize metadata extractor
        self.components["metadata_extractor"] = MetadataExtractor(
            self.components["claude_client"]
        )

        # Initialize organization engine
        org_config = self.config.get("organization", {})
        self.components["organization_engine"] = OrganizationEngine(
            rules_path=org_config.get("rules_path"),
            use_default_rules=org_config.get("use_default_rules", True),
            default_folder=org_config.get("default_folder", "Unsorted"),
        )

        # Initialize file manipulator
        self.components["file_manipulator"] = FileManipulator(
            base_directory=org_config.get("base_directory", "./organized"),
            dry_run=self.config.get("dry_run", False),
        )

        # Initialize error handler
        self.components["error_handler"] = ErrorHandler(
            max_retries=processing_config.get("max_retries", 3)
        )

        # Initialize batch processor
        self.components["batch_processor"] = SimpleBatchProcessor(
            claude_client=self.components["claude_client"],
            file_processor=self.components["file_processor"],
            metadata_extractor=self.components["metadata_extractor"],
            rate_limit=api_config.get("rate_limit", 10),
        )

    def run(
        self,
        source_override: Optional[str] = None,
        target_override: Optional[str] = None,
        dry_run_override: Optional[bool] = None,
    ) -> bool:
        """Run the main application workflow.

        Args:
            source_override: Override source directory
            target_override: Override target directory
            dry_run_override: Override dry-run mode

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_initialized:
            self.initialize()

        # Apply overrides
        if source_override:
            self.config["source"]["directory"] = source_override
            # Reinitialize file accessor
            self.components["file_accessor"] = FileSystemAccessor(source_override)

        if target_override:
            self.config["organization"]["base_directory"] = target_override
            # Reinitialize file manipulator
            self.components["file_manipulator"] = FileManipulator(
                base_directory=target_override,
                dry_run=(
                    dry_run_override
                    if dry_run_override is not None
                    else self.config.get("dry_run", False)
                ),
            )

        if dry_run_override is not None:
            if self.config is None:
                self.config = {}
            self.config["dry_run"] = dry_run_override
            if "file_manipulator" in self.components:
                self.components["file_manipulator"].dry_run = dry_run_override

        try:
            # Step 1: Scan source directory
            logger.info("Scanning source directory for files...")
            files = self._scan_files()

            if not files:
                logger.warning("No files found to process")
                return True

            logger.info(f"Found {len(files)} files to process")

            # Step 2: Initialize progress tracking
            self.progress_tracker = ProgressTracker(len(files))

            # Step 3: Process files in batches
            logger.info("Starting batch processing...")
            processed_files = self._process_files_in_batches(files)

            # Step 4: Organize files based on metadata
            logger.info("Organizing files...")
            organized_results = self._organize_files(processed_files)

            # Step 5: Generate report
            logger.info("Generating report...")
            report = self._generate_report(organized_results)

            # Display summary
            logger.info("\nProcessing complete!")
            logger.info(f"Total files: {len(files)}")
            logger.info(
                f"Successfully processed: {self.progress_tracker.success_count}"
            )
            logger.info(f"Errors: {self.progress_tracker.error_count}")

            if self.config.get("dry_run"):
                logger.info("DRY RUN MODE: No files were actually moved or copied")

            return True

        except Exception as e:
            logger.error(f"Error in main application flow: {str(e)}")
            traceback.print_exc()
            return False

        finally:
            # Cleanup
            self._cleanup()

    def _scan_files(self) -> List[Any]:
        """Scan source directory for files to process."""
        file_accessor = self.components["file_accessor"]
        processing_config = self.config.get("processing", {})

        # Get supported file types
        file_types = processing_config.get("file_types", [])

        if isinstance(file_accessor, DropboxAccessor):
            # Scan Dropbox
            source_folder = self.config.get("dropbox", {}).get("source_folder", "")
            return file_accessor.list_files(source_folder, file_types=file_types)
        else:
            # Scan local filesystem
            files = file_accessor.get_supported_files()

            # Filter by size if configured
            max_size = processing_config.get("max_file_size")
            if max_size:
                files = [f for f in files if f.size <= max_size]

            return files

    def _process_files_in_batches(self, files: List[Any]) -> List[Dict[str, Any]]:
        """Process files in batches using the batch processor."""
        batch_processor = self.components["batch_processor"]
        processing_config = self.config.get("processing", {})
        batch_size = processing_config.get("batch_size", 10)

        processed_results = []
        total_files = len(files)

        # Progress callback
        def progress_callback(current: int, total: int):
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    f"batch_progress",
                    "processing",
                    {"current": current, "total": total},
                )

        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch = files[i : i + batch_size]

            # Convert files to paths
            file_paths = []
            file_map = {}  # Map paths back to original files

            for file in batch:
                if hasattr(file, "path"):  # Local file
                    file_paths.append(file.path)
                    file_map[file.path] = file
                else:  # Dropbox file
                    # Download to temp location first
                    temp_path = self._download_dropbox_file(file)
                    file_paths.append(temp_path)
                    file_map[temp_path] = file

            # Process batch
            batch_results = batch_processor.process_batch(file_paths, progress_callback)

            # Process results
            for file_path, result, error in batch_results:
                original_file = file_map.get(file_path)

                if not original_file:
                    continue

                if error:
                    self.progress_tracker.update_progress(
                        str(original_file), "error", error
                    )
                    processed_results.append(
                        {
                            "file": original_file,
                            "file_path": file_path,
                            "success": False,
                            "error": error,
                        }
                    )
                else:
                    self.progress_tracker.update_progress(
                        str(original_file), "processed", "Successfully processed"
                    )
                    processed_results.append(
                        {
                            "file": original_file,
                            "file_path": file_path,
                            "success": True,
                            "result": result,
                            "metadata": result.get("metadata"),
                        }
                    )

        return processed_results

    def _download_dropbox_file(self, dropbox_file) -> str:
        """Download a Dropbox file to temporary location."""
        import tempfile

        file_accessor = self.components["file_accessor"]
        temp_dir = tempfile.mkdtemp(prefix="file_organizer_")
        temp_path = Path(temp_dir) / dropbox_file.name

        file_accessor.download_file(dropbox_file.path, temp_path)
        return str(temp_path)

    def _organize_files(
        self, processed_files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Organize files based on extracted metadata."""
        organization_engine = self.components["organization_engine"]
        file_manipulator = self.components["file_manipulator"]
        file_accessor = self.components["file_accessor"]

        organized_results = []
        org_config = self.config.get("organization", {})
        mode = org_config.get("mode", "copy")

        for processed in processed_files:
            try:
                file = processed["file"]
                success = processed.get("success", False)

                if not success:
                    # Skip failed files
                    organized_results.append(
                        {
                            "file": file,
                            "source": file.path if hasattr(file, "path") else file.name,
                            "error": processed.get("error", "Processing failed"),
                            "success": False,
                        }
                    )
                    continue

                # Get metadata from processed results
                metadata = processed.get("metadata")

                if not metadata:
                    # Skip files without metadata
                    organized_results.append(
                        {
                            "file": file,
                            "source": file.path if hasattr(file, "path") else file.name,
                            "error": "No metadata available",
                            "success": False,
                        }
                    )
                    continue

                # Determine target location using organization engine
                target_path, rule_name = organization_engine.determine_target_location(
                    metadata
                )

                # Execute file operation
                success = False
                if isinstance(file_accessor, DropboxAccessor):
                    # For Dropbox, use Dropbox API to organize
                    target_dropbox_path = f"{org_config.get('base_directory', '/organized')}/{target_path}"

                    if mode == "move":
                        success = file_accessor.move_file(
                            file.path, target_dropbox_path
                        )
                    else:  # copy
                        success = file_accessor.copy_file(
                            file.path, target_dropbox_path
                        )
                else:
                    # For local files, use file manipulator
                    source_path = (
                        file.path if hasattr(file, "path") else processed["file_path"]
                    )

                    # Use organize_file method which handles both move and copy
                    success = file_manipulator.organize_file(
                        source_path, target_path, mode
                    )

                if success:
                    self.progress_tracker.update_progress(
                        str(file), "success", f"Organized to: {target_path}"
                    )
                    organized_results.append(
                        {
                            "file": file,
                            "source": file.path if hasattr(file, "path") else file.name,
                            "target": target_path,
                            "metadata": metadata,
                            "success": True,
                        }
                    )
                else:
                    self.progress_tracker.update_progress(
                        str(file), "error", "File operation failed"
                    )
                    organized_results.append(
                        {
                            "file": file,
                            "source": file.path if hasattr(file, "path") else file.name,
                            "error": "File operation failed",
                            "success": False,
                        }
                    )

            except Exception as e:
                logger.error(f"Error organizing file {file}: {str(e)}")
                self.progress_tracker.update_progress(str(file), "error", str(e))
                organized_results.append(
                    {
                        "file": file,
                        "source": file.path if hasattr(file, "path") else file.name,
                        "error": str(e),
                        "success": False,
                    }
                )

        return organized_results

    def _generate_report(self, organized_results: List[Dict[str, Any]]) -> str:
        """Generate a summary report of the organization process."""
        report_generator = ReportGenerator(
            self.progress_tracker, self.components["file_manipulator"]
        )

        # Generate report
        report = report_generator.generate_summary_report()

        # Save report to file
        report_dir = (
            Path(
                self.config.get("organization", {}).get("base_directory", "./organized")
            )
            / "reports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"organization_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write(report)

        # Save detailed JSON report
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.progress_tracker.get_summary(),
            "results": organized_results,
            "config": self.config,
        }

        json_file = report_dir / f"organization_report_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(json_report, f, indent=2, default=str)

        logger.info(f"Reports saved to: {report_dir}")
        return report

    def _cleanup(self):
        """Clean up temporary files and resources."""
        # Clean up temporary downloads
        import tempfile
        import shutil

        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            if item.startswith("file_organizer_"):
                try:
                    shutil.rmtree(os.path.join(temp_dir, item))
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Organize files using Claude AI for intelligent categorization"
    )

    parser.add_argument("--config", help="Path to configuration file", default=None)

    parser.add_argument(
        "--source", help="Source directory or Dropbox path", default=None
    )

    parser.add_argument(
        "--target", help="Target directory for organized files", default=None
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without moving files"
    )

    parser.add_argument(
        "--mode", choices=["copy", "move"], help="File operation mode", default=None
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
        default="INFO",
    )

    return parser


def main():
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=getattr(logging, args.log_level))

    # Create and run the application
    app = FileOrganizerApp(config_file=args.config)

    try:
        # Initialize with command-line overrides
        app.initialize()

        # Override configuration with command-line arguments
        if args.mode:
            app.config["organization"]["mode"] = args.mode

        # Run the application
        success = app.run(
            source_override=args.source,
            target_override=args.target,
            dry_run_override=args.dry_run,
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
