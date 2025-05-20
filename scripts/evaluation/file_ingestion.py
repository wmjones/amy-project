#!/usr/bin/env python3
"""File Ingestion and Validation Component for OCR Pipeline."""

import os
import json
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, asdict
import logging
from queue import Queue, PriorityQueue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import magic
from PIL import Image
import PyPDF2
import dropbox
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Container for file metadata."""

    file_id: str
    filename: str
    path: str
    size: int
    file_type: str
    mime_type: str
    hash: str
    created_at: datetime
    modified_at: datetime
    source: str  # 'local' or 'dropbox'
    is_valid: bool
    validation_errors: List[str]
    dimensions: Optional[Tuple[int, int]] = None
    page_count: Optional[int] = None


class FileValidator:
    """Validates files for OCR processing."""

    def __init__(self):
        self.supported_types = {
            "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
            "document": [".pdf", ".docx"],
        }
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.min_resolution = (100, 100)
        self.max_resolution = (10000, 10000)

    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate file for OCR processing."""
        errors = []

        # Check if file exists
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            errors.append(
                f"File too large: {file_size} bytes (max: {self.max_file_size})"
            )

        # Check file extension
        extension = file_path.suffix.lower()
        is_supported = False

        for file_type, extensions in self.supported_types.items():
            if extension in extensions:
                is_supported = True

                # Type-specific validation
                if file_type == "image":
                    img_errors = self._validate_image(file_path)
                    errors.extend(img_errors)
                elif file_type == "document" and extension == ".pdf":
                    pdf_errors = self._validate_pdf(file_path)
                    errors.extend(pdf_errors)

                break

        if not is_supported:
            errors.append(f"Unsupported file type: {extension}")

        # Check file integrity
        try:
            with open(file_path, "rb") as f:
                # Try to read first few bytes
                f.read(1024)
        except Exception as e:
            errors.append(f"File integrity check failed: {e}")

        return len(errors) == 0, errors

    def _validate_image(self, file_path: Path) -> List[str]:
        """Validate image file."""
        errors = []

        try:
            with Image.open(file_path) as img:
                width, height = img.size

                # Check resolution
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    errors.append(f"Image resolution too low: {width}x{height}")
                elif width > self.max_resolution[0] or height > self.max_resolution[1]:
                    errors.append(f"Image resolution too high: {width}x{height}")

                # Check if image is corrupt
                try:
                    img.verify()
                except Exception as e:
                    errors.append(f"Image verification failed: {e}")

        except Exception as e:
            errors.append(f"Failed to open image: {e}")

        return errors

    def _validate_pdf(self, file_path: Path) -> List[str]:
        """Validate PDF file."""
        errors = []

        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    errors.append("PDF is encrypted")

                # Check page count
                page_count = len(pdf_reader.pages)
                if page_count == 0:
                    errors.append("PDF has no pages")
                elif page_count > 1000:
                    errors.append(f"PDF has too many pages: {page_count}")

                # Try to read first page
                try:
                    first_page = pdf_reader.pages[0]
                    _ = first_page.extract_text()
                except Exception as e:
                    errors.append(f"Failed to read PDF content: {e}")

        except Exception as e:
            errors.append(f"Failed to open PDF: {e}")

        return errors


class FileIngestionEngine:
    """Main file ingestion and queuing system."""

    def __init__(self, output_dir: Path, batch_size: int = 10):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.validator = FileValidator()
        self.file_queue = PriorityQueue()
        self.processed_files = set()
        self.metadata_store = {}
        self.lock = threading.Lock()

        # Create output directories
        self.queue_dir = self.output_dir / "queue"
        self.metadata_dir = self.output_dir / "metadata"
        self.failed_dir = self.output_dir / "failed"

        for dir_path in [self.queue_dir, self.metadata_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata from disk."""
        metadata_file = self.metadata_dir / "ingestion_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata_store = json.load(f)
                    self.processed_files = set(self.metadata_store.keys())
                logger.info(f"Loaded metadata for {len(self.processed_files)} files")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_file = self.metadata_dir / "ingestion_metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(self.metadata_store, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _get_file_metadata(
        self, file_path: Path, source: str = "local"
    ) -> FileMetadata:
        """Extract metadata from file."""
        stat = file_path.stat()

        # Get MIME type
        try:
            mime = magic.from_file(str(file_path), mime=True)
        except:
            mime = mimetypes.guess_type(str(file_path))[0] or "unknown"

        # Calculate hash
        file_hash = self._calculate_file_hash(file_path)

        # Create unique ID
        file_id = f"{source}_{file_hash[:16]}_{file_path.name}"

        # Validate file
        is_valid, validation_errors = self.validator.validate_file(file_path)

        # Get dimensions for images
        dimensions = None
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            try:
                with Image.open(file_path) as img:
                    dimensions = img.size
            except:
                pass

        # Get page count for PDFs
        page_count = None
        if file_path.suffix.lower() == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    page_count = len(pdf_reader.pages)
            except:
                pass

        return FileMetadata(
            file_id=file_id,
            filename=file_path.name,
            path=str(file_path),
            size=stat.st_size,
            file_type=file_path.suffix.lower(),
            mime_type=mime,
            hash=file_hash,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            source=source,
            is_valid=is_valid,
            validation_errors=validation_errors,
            dimensions=dimensions,
            page_count=page_count,
        )

    def ingest_local_directory(
        self, directory: Path, recursive: bool = True
    ) -> Dict[str, Any]:
        """Ingest files from local directory."""
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        pattern = "**/*" if recursive else "*"
        file_paths = []

        # Collect all files
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_paths.append(file_path)

        logger.info(f"Found {len(file_paths)} files in {directory}")

        # Process files
        results = {
            "total_files": len(file_paths),
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        with tqdm(total=len(file_paths), desc="Ingesting files") as pbar:
            for file_path in file_paths:
                try:
                    # Get metadata
                    metadata = self._get_file_metadata(file_path)

                    # Check if already processed
                    if metadata.file_id in self.processed_files:
                        results["skipped"] += 1
                        pbar.update(1)
                        continue

                    # Add to queue if valid
                    if metadata.is_valid:
                        # Priority based on file size (smaller files first)
                        priority = metadata.size
                        self.file_queue.put((priority, metadata))

                        # Store metadata
                        with self.lock:
                            self.metadata_store[metadata.file_id] = asdict(metadata)
                            self.processed_files.add(metadata.file_id)

                        results["ingested"] += 1
                    else:
                        # Move to failed directory
                        failed_path = self.failed_dir / file_path.name
                        failed_path.write_bytes(file_path.read_bytes())

                        results["failed"] += 1
                        results["errors"].append(
                            {
                                "file": file_path.name,
                                "errors": metadata.validation_errors,
                            }
                        )

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    results["failed"] += 1
                    results["errors"].append({"file": file_path.name, "error": str(e)})
                    pbar.update(1)

        # Save metadata
        self._save_metadata()

        return results

    def ingest_dropbox_folder(
        self,
        dropbox_client: dropbox.Dropbox,
        folder_path: str,
        download_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Ingest files from Dropbox folder."""
        if download_dir is None:
            download_dir = self.queue_dir / "dropbox_downloads"

        download_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "total_files": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        try:
            # List files in Dropbox folder
            file_list = []
            result = dropbox_client.files_list_folder(folder_path)

            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        file_list.append(entry)

                if not result.has_more:
                    break

                result = dropbox_client.files_list_folder_continue(result.cursor)

            results["total_files"] = len(file_list)
            logger.info(f"Found {len(file_list)} files in Dropbox folder {folder_path}")

            # Download and ingest files
            with tqdm(total=len(file_list), desc="Downloading from Dropbox") as pbar:
                for file_metadata in file_list:
                    try:
                        # Download file
                        local_path = download_dir / file_metadata.name

                        # Check if already downloaded
                        file_id = f"dropbox_{file_metadata.id}_{file_metadata.name}"
                        if file_id in self.processed_files:
                            results["skipped"] += 1
                            pbar.update(1)
                            continue

                        # Download file
                        with open(local_path, "wb") as f:
                            _, response = dropbox_client.files_download(
                                file_metadata.path_display
                            )
                            f.write(response.content)

                        # Get metadata
                        metadata = self._get_file_metadata(local_path, source="dropbox")
                        metadata.file_id = file_id  # Use Dropbox ID

                        # Add to queue if valid
                        if metadata.is_valid:
                            priority = metadata.size
                            self.file_queue.put((priority, metadata))

                            with self.lock:
                                self.metadata_store[metadata.file_id] = asdict(metadata)
                                self.processed_files.add(metadata.file_id)

                            results["ingested"] += 1
                        else:
                            local_path.unlink()  # Remove invalid file
                            results["failed"] += 1
                            results["errors"].append(
                                {
                                    "file": file_metadata.name,
                                    "errors": metadata.validation_errors,
                                }
                            )

                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Failed to download {file_metadata.name}: {e}")
                        results["failed"] += 1
                        results["errors"].append(
                            {"file": file_metadata.name, "error": str(e)}
                        )
                        pbar.update(1)

        except Exception as e:
            logger.error(f"Failed to access Dropbox folder {folder_path}: {e}")
            results["errors"].append({"folder": folder_path, "error": str(e)})

        # Save metadata
        self._save_metadata()

        return results

    def get_next_batch(self) -> List[FileMetadata]:
        """Get next batch of files for processing."""
        batch = []

        while len(batch) < self.batch_size and not self.file_queue.empty():
            try:
                _, metadata = self.file_queue.get_nowait()
                batch.append(metadata)
            except:
                break

        return batch

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self.file_queue.qsize(),
            "processed_count": len(self.processed_files),
            "batch_size": self.batch_size,
            "failed_count": len(list(self.failed_dir.glob("*"))),
        }

    def mark_processed(
        self, file_id: str, success: bool = True, error_message: Optional[str] = None
    ):
        """Mark file as processed."""
        if file_id in self.metadata_store:
            self.metadata_store[file_id]["processed"] = True
            self.metadata_store[file_id]["processed_at"] = datetime.now().isoformat()
            self.metadata_store[file_id]["process_success"] = success

            if error_message:
                self.metadata_store[file_id]["process_error"] = error_message

            self._save_metadata()


def test_ingestion_system():
    """Test the file ingestion system."""

    # Initialize ingestion engine
    output_dir = Path("/workspaces/amy-project/ingestion_test")
    engine = FileIngestionEngine(output_dir, batch_size=5)

    # Test local directory ingestion
    print("Testing local directory ingestion...")
    local_results = engine.ingest_local_directory(
        Path("/workspaces/amy-project/hansman_organized/downloads"), recursive=False
    )

    print(f"\nLocal Ingestion Results:")
    print(f"Total files: {local_results['total_files']}")
    print(f"Ingested: {local_results['ingested']}")
    print(f"Skipped: {local_results['skipped']}")
    print(f"Failed: {local_results['failed']}")

    # Get queue status
    status = engine.get_queue_status()
    print(f"\nQueue Status:")
    print(f"Queue size: {status['queue_size']}")
    print(f"Processed count: {status['processed_count']}")

    # Get next batch
    batch = engine.get_next_batch()
    print(f"\nNext batch ({len(batch)} files):")
    for metadata in batch:
        print(f"  - {metadata.filename} ({metadata.size} bytes)")

    # Save metadata for processed files
    for metadata in batch:
        engine.mark_processed(metadata.file_id, success=True)

    return engine


if __name__ == "__main__":
    test_ingestion_system()
