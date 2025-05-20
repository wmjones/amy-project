"""
Storage system for document metadata.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import shutil

from .extractor import DocumentMetadata

logger = logging.getLogger(__name__)


class MetadataStorage:
    """Store and retrieve document metadata."""

    def __init__(self, storage_path: str, use_database: bool = True):
        """Initialize metadata storage.

        Args:
            storage_path: Path to storage directory
            use_database: Whether to use SQLite database (vs JSON files)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_database = use_database

        if use_database:
            self.db_path = self.storage_path / "metadata.db"
            self._init_database()
        else:
            self.metadata_dir = self.storage_path / "metadata"
            self.metadata_dir.mkdir(exist_ok=True)
            self.index_file = self.storage_path / "index.json"
            self._init_index()

        logger.info(f"Metadata storage initialized at {storage_path}")

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                file_path TEXT PRIMARY KEY,
                document_type TEXT,
                categories TEXT,  -- JSON array
                document_date TEXT,
                entities TEXT,  -- JSON array
                topics TEXT,  -- JSON array
                tags TEXT,  -- JSON array
                summary TEXT,
                suggested_folder TEXT,
                confidence_score REAL,
                processing_timestamp TEXT,
                extracted_text TEXT,
                file_size INTEGER,
                page_count INTEGER,
                language TEXT,
                full_metadata TEXT  -- Complete JSON
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_type ON metadata(document_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_date ON metadata(document_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggested_folder ON metadata(suggested_folder)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processing_timestamp ON metadata(processing_timestamp)"
        )

        conn.commit()
        conn.close()

    def _init_index(self):
        """Initialize JSON index file."""
        if not self.index_file.exists():
            self._save_index({})

    def save_metadata(self, metadata: DocumentMetadata) -> bool:
        """Save document metadata.

        Args:
            metadata: DocumentMetadata object to save

        Returns:
            True if saved successfully
        """
        try:
            if self.use_database:
                return self._save_to_database(metadata)
            else:
                return self._save_to_json(metadata)

        except Exception as e:
            logger.error(f"Error saving metadata for {metadata.source_file}: {e}")
            return False

    def _save_to_database(self, metadata: DocumentMetadata) -> bool:
        """Save metadata to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Convert lists to JSON strings
            metadata_dict = metadata.to_dict()

            cursor.execute(
                """
                INSERT OR REPLACE INTO metadata
                (file_path, document_type, categories, document_date, entities,
                 topics, tags, summary, suggested_folder, confidence_score,
                 processing_timestamp, extracted_text, file_size, page_count,
                 language, full_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.source_file,
                    metadata.document_type,
                    json.dumps(metadata.categories),
                    metadata.dates.document_date,
                    json.dumps([e.__dict__ for e in metadata.entities]),
                    json.dumps(metadata.topics),
                    json.dumps(metadata.tags),
                    metadata.summary,
                    metadata.suggested_folder,
                    metadata.confidence_score,
                    metadata.processing_timestamp,
                    metadata.extracted_text,
                    metadata.file_size,
                    metadata.page_count,
                    metadata.language,
                    json.dumps(metadata_dict),
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def _save_to_json(self, metadata: DocumentMetadata) -> bool:
        """Save metadata to JSON file."""
        try:
            # Generate filename from source file
            source_path = Path(metadata.source_file)
            metadata_filename = f"{source_path.stem}_metadata.json"
            metadata_path = self.metadata_dir / metadata_filename

            # Save metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Update index
            index = self._load_index()
            index[metadata.source_file] = str(metadata_path)
            self._save_index(index)

            return True

        except Exception as e:
            logger.error(f"JSON storage error: {e}")
            return False

    def get_metadata(self, file_path: str) -> Optional[DocumentMetadata]:
        """Retrieve metadata for a file.

        Args:
            file_path: Path to the source file

        Returns:
            DocumentMetadata object or None if not found
        """
        try:
            if self.use_database:
                return self._get_from_database(file_path)
            else:
                return self._get_from_json(file_path)

        except Exception as e:
            logger.error(f"Error retrieving metadata for {file_path}: {e}")
            return None

    def _get_from_database(self, file_path: str) -> Optional[DocumentMetadata]:
        """Get metadata from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT full_metadata FROM metadata WHERE file_path = ?", (file_path,)
            )

            result = cursor.fetchone()
            if result:
                metadata_dict = json.loads(result[0])
                return self._dict_to_metadata(metadata_dict)

            return None

        finally:
            conn.close()

    def _get_from_json(self, file_path: str) -> Optional[DocumentMetadata]:
        """Get metadata from JSON file."""
        try:
            index = self._load_index()

            if file_path in index:
                metadata_path = Path(index[file_path])

                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata_dict = json.load(f)

                    return self._dict_to_metadata(metadata_dict)

            return None

        except Exception as e:
            logger.error(f"Error loading JSON metadata: {e}")
            return None

    def search_metadata(self, **kwargs) -> List[DocumentMetadata]:
        """Search for metadata matching criteria.

        Args:
            **kwargs: Search criteria (document_type, category, date_range, etc.)

        Returns:
            List of matching DocumentMetadata objects
        """
        if self.use_database:
            return self._search_database(**kwargs)
        else:
            return self._search_json(**kwargs)

    def _search_database(self, **kwargs) -> List[DocumentMetadata]:
        """Search database for matching metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Build query
            conditions = []
            params = []

            if "document_type" in kwargs:
                conditions.append("document_type = ?")
                params.append(kwargs["document_type"])

            if "category" in kwargs:
                conditions.append("categories LIKE ?")
                params.append(f'%"{kwargs["category"]}"%')

            if "date_after" in kwargs:
                conditions.append("document_date >= ?")
                params.append(kwargs["date_after"])

            if "date_before" in kwargs:
                conditions.append("document_date <= ?")
                params.append(kwargs["date_before"])

            if "min_confidence" in kwargs:
                conditions.append("confidence_score >= ?")
                params.append(kwargs["min_confidence"])

            # Build and execute query
            query = "SELECT full_metadata FROM metadata"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            cursor.execute(query, params)

            # Convert results
            results = []
            for row in cursor.fetchall():
                metadata_dict = json.loads(row[0])
                metadata = self._dict_to_metadata(metadata_dict)
                if metadata:
                    results.append(metadata)

            return results

        finally:
            conn.close()

    def _search_json(self, **kwargs) -> List[DocumentMetadata]:
        """Search JSON files for matching metadata."""
        results = []
        index = self._load_index()

        for file_path, metadata_path in index.items():
            metadata = self._get_from_json(file_path)

            if metadata and self._matches_criteria(metadata, **kwargs):
                results.append(metadata)

        return results

    def _matches_criteria(self, metadata: DocumentMetadata, **kwargs) -> bool:
        """Check if metadata matches search criteria."""
        if (
            "document_type" in kwargs
            and metadata.document_type != kwargs["document_type"]
        ):
            return False

        if "category" in kwargs and kwargs["category"] not in metadata.categories:
            return False

        if "date_after" in kwargs and metadata.dates.document_date:
            if metadata.dates.document_date < kwargs["date_after"]:
                return False

        if "date_before" in kwargs and metadata.dates.document_date:
            if metadata.dates.document_date > kwargs["date_before"]:
                return False

        if (
            "min_confidence" in kwargs
            and metadata.confidence_score < kwargs["min_confidence"]
        ):
            return False

        return True

    def export_metadata(self, output_path: str, format: str = "json") -> bool:
        """Export all metadata to file.

        Args:
            output_path: Path for output file
            format: Export format ('json', 'csv')

        Returns:
            True if exported successfully
        """
        try:
            if format == "json":
                return self._export_json(output_path)
            elif format == "csv":
                return self._export_csv(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            return False

    def _export_json(self, output_path: str) -> bool:
        """Export metadata to JSON file."""
        all_metadata = []

        if self.use_database:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("SELECT full_metadata FROM metadata")

                for row in cursor.fetchall():
                    metadata_dict = json.loads(row[0])
                    all_metadata.append(metadata_dict)

            finally:
                conn.close()
        else:
            index = self._load_index()

            for file_path in index:
                metadata = self.get_metadata(file_path)
                if metadata:
                    all_metadata.append(metadata.to_dict())

        # Write to file
        with open(output_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        return True

    def _export_csv(self, output_path: str) -> bool:
        """Export metadata to CSV file."""
        import csv

        # Get all metadata
        all_metadata = []

        if self.use_database:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    SELECT file_path, document_type, categories, document_date,
                           summary, suggested_folder, confidence_score,
                           processing_timestamp
                    FROM metadata
                """
                )

                headers = [
                    "file_path",
                    "document_type",
                    "categories",
                    "document_date",
                    "summary",
                    "suggested_folder",
                    "confidence_score",
                    "processing_timestamp",
                ]

                rows = cursor.fetchall()

            finally:
                conn.close()
        else:
            # Collect from JSON files
            headers = [
                "file_path",
                "document_type",
                "categories",
                "document_date",
                "summary",
                "suggested_folder",
                "confidence_score",
                "processing_timestamp",
            ]
            rows = []

            index = self._load_index()
            for file_path in index:
                metadata = self.get_metadata(file_path)
                if metadata:
                    rows.append(
                        [
                            metadata.source_file,
                            metadata.document_type,
                            ", ".join(metadata.categories),
                            metadata.dates.document_date or "",
                            metadata.summary,
                            metadata.suggested_folder,
                            metadata.confidence_score,
                            metadata.processing_timestamp,
                        ]
                    )

        # Write CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        return True

    def _dict_to_metadata(self, data: Dict[str, Any]) -> DocumentMetadata:
        """Convert dictionary to DocumentMetadata object."""
        from .extractor import DateInfo, Entity

        # Parse dates
        dates_data = data.get("dates", {})
        dates = DateInfo(
            document_date=dates_data.get("document_date"),
            mentioned_dates=dates_data.get("mentioned_dates", []),
        )

        # Parse entities
        entities = []
        for entity_data in data.get("entities", []):
            entities.append(
                Entity(
                    name=entity_data["name"],
                    type=entity_data["type"],
                    confidence=entity_data.get("confidence", 0.8),
                )
            )

        # Create metadata object
        return DocumentMetadata(
            document_type=data["document_type"],
            categories=data["categories"],
            dates=dates,
            entities=entities,
            topics=data.get("topics", []),
            tags=data.get("tags", []),
            summary=data.get("summary", ""),
            suggested_folder=data["suggested_folder"],
            confidence_score=data["confidence_score"],
            source_file=data["source_file"],
            processing_timestamp=data["processing_timestamp"],
            extracted_text=data.get("extracted_text"),
            file_size=data.get("file_size"),
            page_count=data.get("page_count"),
            language=data.get("language", "en"),
        )

    def _load_index(self) -> Dict[str, str]:
        """Load JSON index file."""
        try:
            with open(self.index_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: Dict[str, str]):
        """Save JSON index file."""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    def backup(self, backup_path: str) -> bool:
        """Create backup of metadata storage.

        Args:
            backup_path: Path for backup

        Returns:
            True if backup successful
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.use_database:
                # Backup database
                backup_file = backup_dir / f"metadata_backup_{timestamp}.db"
                shutil.copy2(self.db_path, backup_file)
            else:
                # Backup JSON files
                backup_subdir = backup_dir / f"metadata_backup_{timestamp}"
                shutil.copytree(self.storage_path, backup_subdir)

            logger.info(f"Backup created at {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
