"""
Metadata extraction from Claude API responses.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import re
from pathlib import Path

from src.claude_integration.client import ClaudeClient, AnalysisResult
from src.claude_integration.prompts import PromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity extracted from document."""

    name: str
    type: str  # person, organization, location, etc.
    confidence: float = 0.8


@dataclass
class DateInfo:
    """Date information extracted from document."""

    document_date: Optional[str] = None  # Primary document date
    mentioned_dates: List[str] = field(default_factory=list)  # Other dates mentioned
    date_format: str = "ISO-8601"  # Format used


@dataclass
class DocumentMetadata:
    """Complete metadata for a document."""

    document_type: str
    categories: List[str]
    dates: DateInfo
    entities: List[Entity]
    topics: List[str]
    tags: List[str]
    summary: str
    suggested_folder: str
    confidence_score: float
    source_file: str
    processing_timestamp: str
    extracted_text: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        # Ensure dates is properly serialized
        data["dates"] = {
            "document_date": self.dates.document_date,
            "mentioned_dates": self.dates.mentioned_dates,
            "date_format": self.dates.date_format,
        }
        # Ensure entities are properly serialized
        data["entities"] = [asdict(e) for e in self.entities]
        return data


class MetadataExtractor:
    """Extract and process metadata from documents."""

    def __init__(self, claude_client: ClaudeClient, schema_path: Optional[str] = None):
        """Initialize metadata extractor.

        Args:
            claude_client: Claude API client instance
            schema_path: Optional path to custom schema file
        """
        self.claude_client = claude_client
        self.schema = self._load_schema(schema_path)
        self.date_patterns = [
            # ISO format
            r"\d{4}-\d{2}-\d{2}",
            # US format
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            # European format
            r"\d{1,2}\.\d{1,2}\.\d{2,4}",
            # Long format
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            # Short format
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
        ]

        logger.info("Metadata extractor initialized")

    def extract_metadata(
        self,
        file_content: str,
        file_path: str,
        file_type: str,
        file_size: Optional[int] = None,
        analysis_result: Optional[AnalysisResult] = None,
    ) -> DocumentMetadata:
        """Extract metadata from document content.

        Args:
            file_content: Content of the file (text or description)
            file_path: Path to the source file
            file_type: Type of file (pdf, docx, jpg, etc.)
            file_size: Optional file size in bytes
            analysis_result: Optional pre-analyzed result from Claude

        Returns:
            DocumentMetadata object
        """
        try:
            # Use provided analysis result or get new one
            if analysis_result is None:
                # Select appropriate prompt
                prompt_template = PromptTemplates.select_prompt(file_type, file_content)
                formatted_prompt = PromptTemplates.format_prompt(
                    prompt_template,
                    content=file_content[:10000],  # Limit content length
                )

                # Analyze with Claude
                analysis_result = self.claude_client.analyze_document(
                    content=file_content,
                    file_name=Path(file_path).name,
                    file_type=file_type,
                    custom_prompt=formatted_prompt,
                )

            # Parse the response
            parsed_data = self._parse_response(analysis_result)

            # Extract additional metadata from content
            dates = self._extract_dates(file_content, parsed_data)
            entities = self._extract_entities(parsed_data)

            # Normalize and validate
            normalized_data = self._normalize_metadata(parsed_data, dates, entities)

            # Create metadata object
            metadata = DocumentMetadata(
                document_type=normalized_data.get("document_type", "general"),
                categories=normalized_data.get(
                    "categories", [normalized_data.get("category", "uncategorized")]
                ),
                dates=dates,
                entities=entities,
                topics=normalized_data.get(
                    "subjects", normalized_data.get("topics", [])
                ),
                tags=normalized_data.get("tags", []),
                summary=normalized_data.get("summary", ""),
                suggested_folder=normalized_data.get("suggested_folder", "unsorted"),
                confidence_score=normalized_data.get("confidence_score", 0.5),
                source_file=file_path,
                processing_timestamp=datetime.now().isoformat(),
                extracted_text=file_content[:1000] if file_content else None,
                file_size=file_size,
                page_count=normalized_data.get("page_count"),
                language=normalized_data.get("language", "en"),
            )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            # Return minimal metadata on error
            return self._create_fallback_metadata(file_path, file_type, str(e))

    def _parse_response(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Parse Claude's response into structured data."""
        try:
            # Try to parse as JSON first
            if analysis_result.content.strip().startswith("{"):
                return json.loads(analysis_result.content)

            # If not JSON, try to extract structured data
            parsed_data = {"confidence_score": analysis_result.confidence_score}

            # Extract from content using patterns
            content = analysis_result.content

            # Extract document type
            type_match = re.search(r"document[_\s]type[:\s]+([^\n,]+)", content, re.I)
            if type_match:
                parsed_data["document_type"] = type_match.group(1).strip()

            # Extract category
            cat_match = re.search(r"category[:\s]+([^\n,]+)", content, re.I)
            if cat_match:
                parsed_data["category"] = cat_match.group(1).strip()

            # Extract summary
            summary_match = re.search(r"summary[:\s]+([^\n]+)", content, re.I)
            if summary_match:
                parsed_data["summary"] = summary_match.group(1).strip()

            # Extract folder suggestion
            folder_match = re.search(
                r"suggested[_\s]folder[:\s]+([^\n,]+)", content, re.I
            )
            if folder_match:
                parsed_data["suggested_folder"] = folder_match.group(1).strip()

            return parsed_data

        except json.JSONDecodeError:
            logger.warning("Failed to parse response as JSON, using fallback parsing")
            return {"raw_content": analysis_result.content}
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def _extract_dates(self, content: str, parsed_data: Dict[str, Any]) -> DateInfo:
        """Extract dates from content and parsed data."""
        dates = DateInfo()

        # Get document date from parsed data
        if "date" in parsed_data:
            dates.document_date = self._normalize_date(parsed_data["date"])

        # Extract dates from content
        for pattern in self.date_patterns:
            matches = re.findall(pattern, content, re.I)
            for match in matches:
                normalized_date = self._normalize_date(match)
                if normalized_date and normalized_date not in dates.mentioned_dates:
                    dates.mentioned_dates.append(normalized_date)

        # Set document date to first mentioned date if not already set
        if not dates.document_date and dates.mentioned_dates:
            dates.document_date = dates.mentioned_dates[0]

        return dates

    def _extract_entities(self, parsed_data: Dict[str, Any]) -> List[Entity]:
        """Extract entities from parsed data."""
        entities = []

        if "entities" in parsed_data:
            entity_data = parsed_data["entities"]

            # Handle different entity formats
            if isinstance(entity_data, dict):
                for entity_type, names in entity_data.items():
                    if isinstance(names, list):
                        for name in names:
                            entities.append(Entity(name=name, type=entity_type))
                    else:
                        entities.append(Entity(name=str(names), type=entity_type))

            elif isinstance(entity_data, list):
                for item in entity_data:
                    if isinstance(item, dict):
                        entities.append(
                            Entity(
                                name=item.get("name", ""),
                                type=item.get("type", "unknown"),
                            )
                        )
                    else:
                        entities.append(Entity(name=str(item), type="unknown"))

        return entities

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to ISO-8601 format."""
        if not date_str:
            return None

        try:
            # Try parsing common formats
            formats = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%B %d, %Y", "%b %d, %Y"]

            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str.strip(), fmt)
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue

            # If no format matches, try to extract year at least
            year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
            if year_match:
                return f"{year_match.group(0)}-01-01"

            return None

        except Exception as e:
            logger.warning(f"Error normalizing date '{date_str}': {e}")
            return None

    def _normalize_metadata(
        self, parsed_data: Dict[str, Any], dates: DateInfo, entities: List[Entity]
    ) -> Dict[str, Any]:
        """Normalize and validate metadata."""
        normalized = parsed_data.copy()

        # Ensure required fields
        if "document_type" not in normalized or not normalized["document_type"]:
            normalized["document_type"] = "general"

        # Normalize categories
        if "categories" not in normalized:
            if "category" in normalized:
                normalized["categories"] = [normalized["category"]]
            else:
                normalized["categories"] = ["uncategorized"]

        # Ensure lists are lists
        for field in ["subjects", "topics", "tags"]:
            if field in normalized and not isinstance(normalized[field], list):
                normalized[field] = [normalized[field]]

        # Clean suggested folder path
        if "suggested_folder" in normalized:
            folder = normalized["suggested_folder"]
            # Remove invalid characters
            folder = re.sub(r'[<>:"|?*]', "", folder)
            # Replace backslashes with forward slashes
            folder = folder.replace("\\", "/")
            # Remove leading/trailing slashes
            folder = folder.strip("/")
            normalized["suggested_folder"] = folder

        # Validate confidence score
        if "confidence_score" in normalized:
            try:
                score = float(normalized["confidence_score"])
                normalized["confidence_score"] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                normalized["confidence_score"] = 0.5

        return normalized

    def _create_fallback_metadata(
        self, file_path: str, file_type: str, error: str
    ) -> DocumentMetadata:
        """Create fallback metadata when extraction fails."""
        return DocumentMetadata(
            document_type="unknown",
            categories=["error"],
            dates=DateInfo(),
            entities=[],
            topics=[],
            tags=["extraction_failed"],
            summary=f"Metadata extraction failed: {error}",
            suggested_folder="errors",
            confidence_score=0.0,
            source_file=file_path,
            processing_timestamp=datetime.now().isoformat(),
            language="unknown",
        )

    def _load_schema(self, schema_path: Optional[str] = None) -> Dict[str, Any]:
        """Load metadata schema from file or use default."""
        if schema_path and Path(schema_path).exists():
            try:
                with open(schema_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load schema from {schema_path}: {e}")

        # Return default schema
        return {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "categories": {"type": "array", "items": {"type": "string"}},
                "dates": {
                    "type": "object",
                    "properties": {
                        "document_date": {"type": "string", "format": "date"},
                        "mentioned_dates": {
                            "type": "array",
                            "items": {"type": "string", "format": "date"},
                        },
                    },
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                        },
                    },
                },
                "topics": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "suggested_folder": {"type": "string"},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["document_type", "categories", "confidence_score"],
        }

    def validate_metadata(self, metadata: DocumentMetadata) -> bool:
        """Validate metadata against schema."""
        try:
            # Basic validation
            if not metadata.document_type:
                return False

            if not metadata.categories:
                return False

            if metadata.confidence_score < 0 or metadata.confidence_score > 1:
                return False

            # Additional validation can be added here
            return True

        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return False
