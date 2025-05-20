"""
Simplified batch processor for the main application.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.claude_integration.client import ClaudeClient
from src.file_access.processor import FileProcessor
from src.metadata_extraction.extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class SimpleBatchProcessor:
    """Simplified batch processor that processes files without queue management."""

    def __init__(
        self,
        claude_client: ClaudeClient,
        file_processor: FileProcessor,
        metadata_extractor: MetadataExtractor,
        rate_limit: int = 10,
    ):
        """Initialize simple batch processor.

        Args:
            claude_client: Claude API client
            file_processor: File processor
            metadata_extractor: Metadata extractor
            rate_limit: Maximum requests per minute
        """
        self.claude_client = claude_client
        self.file_processor = file_processor
        self.metadata_extractor = metadata_extractor
        self.rate_limit = rate_limit
        self.last_request_time = 0

    def process_batch(
        self, file_paths: List[str], progress_callback=None
    ) -> List[Tuple[str, Dict[str, Any], Optional[str]]]:
        """Process a batch of files.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates

        Returns:
            List of (file_path, result_data, error) tuples
        """
        results = []
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Process the file
                result = self._process_single_file(file_path)
                results.append((file_path, result, None))

                if progress_callback:
                    progress_callback(i + 1, total)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append((file_path, {}, str(e)))

        return results

    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file.

        Args:
            file_path: Path to file

        Returns:
            Processing result dictionary
        """
        # Process file content
        processed = self.file_processor.process_file(file_path)

        if not processed.success:
            raise Exception(f"Failed to process file: {processed.error}")

        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(
            file_content=processed.content,
            file_path=file_path,
            file_type=processed.format,
            file_size=Path(file_path).stat().st_size,
        )

        # Return combined result
        return {
            "metadata": metadata,
            "processed_content": processed.content[:1000],  # First 1000 chars
            "format": processed.format,
            "chunks": processed.chunks,
            "confidence_score": metadata.confidence_score,
        }

    def _apply_rate_limit(self):
        """Apply rate limiting for API calls."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit  # seconds between requests

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()
