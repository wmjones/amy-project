"""
Batch processing system with queue management.
"""

import sqlite3
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading
from queue import Queue, PriorityQueue
import os

from src.claude_integration.client import ClaudeClient
from src.file_access.processor import FileProcessor
from src.metadata_extraction.extractor import MetadataExtractor
from src.organization_logic.engine import OrganizationEngine
from src.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Item in the processing queue."""

    id: Optional[int]
    file_path: str
    status: str  # 'queued', 'processing', 'completed', 'failed', 'paused'
    priority: int
    added_time: float
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BatchProcessor:
    """Batch processor with queue management and rate limiting."""

    def __init__(
        self,
        claude_client: ClaudeClient,
        file_processor: FileProcessor,
        metadata_extractor: MetadataExtractor,
        organization_engine: OrganizationEngine,
        db_path: str = "processing_queue.db",
        batch_size: int = 10,
        rate_limit: int = 60,  # requests per minute
        max_retries: int = 3,
        pause_on_error: bool = False,
    ):
        """Initialize batch processor.

        Args:
            claude_client: Claude API client
            file_processor: File processor
            metadata_extractor: Metadata extractor
            organization_engine: Organization engine
            db_path: Path to SQLite database
            batch_size: Number of files to process in each batch
            rate_limit: Maximum API requests per minute
            max_retries: Maximum retry attempts for failed files
            pause_on_error: Whether to pause processing on errors
        """
        self.claude_client = claude_client
        self.file_processor = file_processor
        self.metadata_extractor = metadata_extractor
        self.organization_engine = organization_engine
        self.db_path = db_path
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.pause_on_error = pause_on_error

        # Rate limiting
        self.last_request_time = 0
        self.rate_limiter_lock = threading.Lock()

        # Processing state
        self.is_paused = False
        self.is_running = False
        self.current_batch = []

        # Statistics
        self.processing_stats = {
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
        }

        # Initialize database
        self._initialize_database()
        self._update_stats()

        logger.info(
            f"BatchProcessor initialized with batch_size={batch_size}, rate_limit={rate_limit}"
        )

    def _initialize_database(self):
        """Initialize SQLite database for queue management."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create queue table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                added_time REAL NOT NULL,
                started_time REAL,
                completed_time REAL,
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON processing_queue(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_priority ON processing_queue(priority DESC, added_time)"
        )

        # Create processing history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                started_time REAL,
                completed_time REAL,
                error TEXT,
                metadata TEXT
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self.db_path}")

    def add_to_queue(
        self,
        file_paths: List[str],
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add files to the processing queue.

        Args:
            file_paths: List of file paths to add
            priority: Priority level (higher = more important)
            metadata: Optional metadata for all files

        Returns:
            Number of files added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        added_count = 0
        current_time = time.time()

        for file_path in file_paths:
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO processing_queue
                    (file_path, status, priority, added_time, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        file_path,
                        "queued",
                        priority,
                        current_time,
                        json.dumps(metadata) if metadata else None,
                    ),
                )

                if cursor.rowcount > 0:
                    added_count += 1

            except sqlite3.IntegrityError:
                logger.warning(f"File already in queue: {file_path}")

        conn.commit()
        conn.close()

        self._update_stats()
        logger.info(f"Added {added_count} files to queue")

        return added_count

    def process_batch(
        self, progress_tracker: Optional[ProgressTracker] = None
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """Process a batch of files.

        Args:
            progress_tracker: Optional progress tracker

        Returns:
            List of (file_path, success, error) tuples
        """
        if self.is_paused:
            logger.info("Processing is paused")
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next batch
        cursor.execute(
            """
            SELECT id, file_path, priority, retry_count, metadata
            FROM processing_queue
            WHERE status IN ('queued', 'failed')
            AND retry_count < ?
            ORDER BY priority DESC, added_time ASC
            LIMIT ?
        """,
            (self.max_retries, self.batch_size),
        )

        batch = cursor.fetchall()

        if not batch:
            logger.info("No files to process")
            conn.close()
            return []

        results = []

        for row in batch:
            file_id, file_path, priority, retry_count, metadata_str = row

            # Update status to processing
            cursor.execute(
                """
                UPDATE processing_queue
                SET status = 'processing', started_time = ?
                WHERE id = ?
            """,
                (time.time(), file_id),
            )
            conn.commit()

            # Process file
            try:
                result = self._process_single_file(
                    file_path,
                    json.loads(metadata_str) if metadata_str else None,
                    progress_tracker,
                )

                # Update status to completed
                cursor.execute(
                    """
                    UPDATE processing_queue
                    SET status = 'completed', completed_time = ?
                    WHERE id = ?
                """,
                    (time.time(), file_id),
                )

                # Add to history
                cursor.execute(
                    """
                    INSERT INTO processing_history
                    (file_path, status, started_time, completed_time)
                    VALUES (?, 'completed', ?, ?)
                """,
                    (file_path, time.time(), time.time()),
                )

                results.append((file_path, True, None))

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing {file_path}: {error_msg}")

                # Update retry count and status
                new_status = (
                    "failed" if retry_count + 1 >= self.max_retries else "queued"
                )
                cursor.execute(
                    """
                    UPDATE processing_queue
                    SET status = ?, error = ?, retry_count = retry_count + 1
                    WHERE id = ?
                """,
                    (new_status, error_msg, file_id),
                )

                # Add to history
                cursor.execute(
                    """
                    INSERT INTO processing_history
                    (file_path, status, started_time, completed_time, error)
                    VALUES (?, 'failed', ?, ?, ?)
                """,
                    (file_path, time.time(), time.time(), error_msg),
                )

                results.append((file_path, False, error_msg))

                if self.pause_on_error:
                    self.pause()
                    logger.warning("Processing paused due to error")
                    break

            conn.commit()

        conn.close()
        self._update_stats()

        return results

    def _process_single_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]],
        progress_tracker: Optional[ProgressTracker],
    ) -> Dict[str, Any]:
        """Process a single file.

        Args:
            file_path: Path to file
            metadata: Optional metadata
            progress_tracker: Optional progress tracker

        Returns:
            Processing result
        """
        # Apply rate limiting
        self._apply_rate_limit()

        # Process file
        processed = self.file_processor.process_file(file_path)

        if not processed.success:
            raise Exception(f"Failed to process file: {processed.error}")

        # Extract metadata with Claude
        analysis = self.metadata_extractor.extract_metadata(
            file_content=processed.content,
            file_path=file_path,
            file_type=processed.format,
            file_size=Path(file_path).stat().st_size,
        )

        # Determine organization
        target_path, rule_used = self.organization_engine.determine_target_location(
            analysis
        )

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update_progress(
                file_path,
                "success",
                details={
                    "target_path": target_path,
                    "rule_used": rule_used,
                    "confidence": analysis.confidence_score,
                },
            )

        return {
            "file_path": file_path,
            "metadata": analysis.to_dict(),
            "target_path": target_path,
            "rule_used": rule_used,
        }

    def _apply_rate_limit(self):
        """Apply rate limiting for API calls."""
        with self.rate_limiter_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            min_interval = 60.0 / self.rate_limit  # seconds between requests

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)

            self.last_request_time = time.time()

    def process_all(
        self,
        progress_tracker: Optional[ProgressTracker] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Process all queued files.

        Args:
            progress_tracker: Optional progress tracker
            callback: Optional callback function called after each batch

        Returns:
            Processing summary
        """
        self.is_running = True
        start_time = time.time()
        total_processed = 0
        total_success = 0
        total_failed = 0

        try:
            while self.is_running and not self.is_paused:
                results = self.process_batch(progress_tracker)

                if not results:
                    break  # No more files to process

                # Update counters
                total_processed += len(results)
                total_success += sum(1 for _, success, _ in results if success)
                total_failed += sum(1 for _, success, _ in results if not success)

                # Call callback if provided
                if callback:
                    callback(results)

                # Check if we should continue
                if self.get_queue_size() == 0:
                    break

        finally:
            self.is_running = False

        end_time = time.time()

        return {
            "total_processed": total_processed,
            "successful": total_success,
            "failed": total_failed,
            "duration": end_time - start_time,
            "files_per_second": (
                total_processed / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
        }

    def pause(self):
        """Pause processing."""
        self.is_paused = True
        logger.info("Processing paused")

    def resume(self):
        """Resume processing."""
        self.is_paused = False
        logger.info("Processing resumed")

    def stop(self):
        """Stop processing."""
        self.is_running = False
        self.is_paused = False
        logger.info("Processing stopped")

    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue status.

        Returns:
            Dictionary with status counts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status, COUNT(*)
            FROM processing_queue
            GROUP BY status
        """
        )

        status_counts = dict(cursor.fetchall())
        conn.close()

        return status_counts

    def get_queue_size(self) -> int:
        """Get number of files in queue.

        Returns:
            Total number of queued files
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM processing_queue
            WHERE status IN ('queued', 'processing')
        """
        )

        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_failed_files(self, limit: int = 100) -> List[QueueItem]:
        """Get failed files from queue.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of failed queue items
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, file_path, status, priority, added_time,
                   started_time, completed_time, error, retry_count, metadata
            FROM processing_queue
            WHERE status = 'failed'
            ORDER BY added_time DESC
            LIMIT ?
        """,
            (limit,),
        )

        items = []
        for row in cursor.fetchall():
            item = QueueItem(
                id=row[0],
                file_path=row[1],
                status=row[2],
                priority=row[3],
                added_time=row[4],
                started_time=row[5],
                completed_time=row[6],
                error=row[7],
                retry_count=row[8],
                metadata=json.loads(row[9]) if row[9] else None,
            )
            items.append(item)

        conn.close()
        return items

    def retry_failed_files(self) -> int:
        """Retry all failed files.

        Returns:
            Number of files queued for retry
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE processing_queue
            SET status = 'queued', retry_count = 0, error = NULL
            WHERE status = 'failed'
        """
        )

        count = cursor.rowcount
        conn.commit()
        conn.close()

        self._update_stats()
        logger.info(f"Queued {count} failed files for retry")

        return count

    def clear_completed(self) -> int:
        """Clear completed files from queue.

        Returns:
            Number of files cleared
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM processing_queue
            WHERE status = 'completed'
        """
        )

        count = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleared {count} completed files from queue")

        return count

    def export_queue_state(self, output_path: str):
        """Export current queue state to file.

        Args:
            output_path: Path to save queue state
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM processing_queue
            ORDER BY priority DESC, added_time
        """
        )

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        queue_state = {
            "export_time": datetime.now().isoformat(),
            "queue_size": len(rows),
            "status_summary": self.get_queue_status(),
            "items": [dict(zip(columns, row)) for row in rows],
        }

        with open(output_path, "w") as f:
            json.dump(queue_state, f, indent=2)

        conn.close()
        logger.info(f"Queue state exported to {output_path}")

    def _update_stats(self):
        """Update processing statistics."""
        status_counts = self.get_queue_status()

        self.processing_stats = {
            "queued": status_counts.get("queued", 0),
            "processing": status_counts.get("processing", 0),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
        }

    def get_processing_history(
        self, limit: int = 100, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get processing history.

        Args:
            limit: Maximum number of records
            status: Filter by status

        Returns:
            List of history records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM processing_history"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY completed_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        history = [dict(zip(columns, row)) for row in rows]

        conn.close()
        return history


class AsyncBatchProcessor(BatchProcessor):
    """Asynchronous batch processor using threading."""

    def __init__(self, *args, **kwargs):
        """Initialize async batch processor."""
        super().__init__(*args, **kwargs)
        self.processing_thread = None
        self.stop_event = threading.Event()

    def start_processing(
        self,
        progress_tracker: Optional[ProgressTracker] = None,
        callback: Optional[Callable] = None,
    ):
        """Start asynchronous processing.

        Args:
            progress_tracker: Optional progress tracker
            callback: Optional callback function
        """
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Processing already in progress")
            return

        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._process_worker, args=(progress_tracker, callback)
        )
        self.processing_thread.start()
        logger.info("Started async processing")

    def stop_processing(self):
        """Stop asynchronous processing."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Stopped async processing")

    def _process_worker(
        self, progress_tracker: Optional[ProgressTracker], callback: Optional[Callable]
    ):
        """Worker thread for processing."""
        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(1)
                continue

            results = self.process_batch(progress_tracker)

            if not results:
                # No more files, wait before checking again
                time.sleep(5)
                continue

            if callback:
                callback(results)
