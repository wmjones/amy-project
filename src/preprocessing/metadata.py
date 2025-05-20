"""
Document metadata profile for tracking processing history.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Represents a single processing step in the pipeline."""

    operation: str
    timestamp: str
    parameters: Dict[str, Any]
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    output_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageCharacteristics:
    """Characteristics of the processed image."""

    width: int
    height: int
    channels: int
    format: str
    dpi: Optional[int] = None
    color_space: str = "RGB"
    bit_depth: int = 8
    file_size_bytes: int = 0
    checksum: str = ""

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get image resolution as tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0


@dataclass
class QualityMetrics:
    """Quality metrics for document processing."""

    sharpness_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    text_clarity_score: float = 0.0
    noise_level: float = 0.0
    skew_angle: float = 0.0
    overall_quality: float = 0.0


@dataclass
class MetadataProfile:
    """Complete metadata profile for a document processing pipeline."""

    document_id: str
    source_path: str
    created_at: str
    updated_at: str
    image_characteristics: ImageCharacteristics
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    document_regions: List[Dict[str, Any]] = field(default_factory=list)
    enhancement_variants: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_processing_step(
        self,
        operation: str,
        parameters: Dict[str, Any],
        duration: float,
        success: bool = True,
        error_message: Optional[str] = None,
        output_info: Optional[Dict[str, Any]] = None,
    ):
        """Add a processing step to the history.

        Args:
            operation: Name of the operation performed
            parameters: Parameters used for the operation
            duration: Duration in seconds
            success: Whether the operation was successful
            error_message: Error message if operation failed
            output_info: Additional output information
        """
        step = ProcessingStep(
            operation=operation,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            output_info=output_info or {},
        )
        self.processing_steps.append(step)
        self.updated_at = datetime.now().isoformat()

    def get_total_processing_time(self) -> float:
        """Calculate total processing time across all steps.

        Returns:
            Total duration in seconds
        """
        return sum(step.duration_seconds for step in self.processing_steps)

    def get_failed_steps(self) -> List[ProcessingStep]:
        """Get list of failed processing steps.

        Returns:
            List of failed steps
        """
        return [step for step in self.processing_steps if not step.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata profile to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert metadata profile to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                if hasattr(obj, "tolist"):  # numpy array
                    return obj.tolist()
                return super().default(obj)

        return json.dumps(self.to_dict(), indent=indent, cls=NumpyEncoder)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataProfile":
        """Create metadata profile from dictionary.

        Args:
            data: Dictionary data

        Returns:
            MetadataProfile instance
        """
        # Convert nested dataclasses
        image_chars = ImageCharacteristics(**data["image_characteristics"])
        processing_steps = [ProcessingStep(**step) for step in data["processing_steps"]]
        quality_metrics = (
            QualityMetrics(**data["quality_metrics"])
            if data.get("quality_metrics")
            else None
        )

        return cls(
            document_id=data["document_id"],
            source_path=data["source_path"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            image_characteristics=image_chars,
            processing_steps=processing_steps,
            quality_metrics=quality_metrics,
            document_regions=data.get("document_regions", []),
            enhancement_variants=data.get("enhancement_variants", {}),
            tags=data.get("tags", []),
            custom_metadata=data.get("custom_metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MetadataProfile":
        """Create metadata profile from JSON string.

        Args:
            json_str: JSON string

        Returns:
            MetadataProfile instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, file_path: str):
        """Save metadata profile to file.

        Args:
            file_path: Path to save the metadata
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.to_json())

        logger.info(f"Saved metadata profile to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str) -> "MetadataProfile":
        """Load metadata profile from file.

        Args:
            file_path: Path to the metadata file

        Returns:
            MetadataProfile instance
        """
        with open(file_path, "r") as f:
            json_str = f.read()

        return cls.from_json(json_str)


def create_metadata_profile(
    image_path: str, document_id: Optional[str] = None
) -> MetadataProfile:
    """Create a new metadata profile for a document.

    Args:
        image_path: Path to the source image
        document_id: Optional document ID (will be generated if not provided)

    Returns:
        New MetadataProfile instance
    """
    image_path = Path(image_path)

    # Generate document ID if not provided
    if not document_id:
        # Use file hash for consistent ID
        with open(image_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        document_id = f"{image_path.stem}_{file_hash}"

    # Get basic image info (would be populated by actual image analysis)
    file_stats = image_path.stat()

    image_characteristics = ImageCharacteristics(
        width=0,  # To be filled by actual image analysis
        height=0,
        channels=3,
        format=image_path.suffix.lstrip("."),
        file_size_bytes=file_stats.st_size,
        checksum=document_id.split("_")[-1] if "_" in document_id else "",
    )

    now = datetime.now().isoformat()

    return MetadataProfile(
        document_id=document_id,
        source_path=str(image_path),
        created_at=now,
        updated_at=now,
        image_characteristics=image_characteristics,
    )


def calculate_quality_metrics(image: Any) -> QualityMetrics:
    """Calculate quality metrics for an image.

    Args:
        image: Image array (numpy array)

    Returns:
        QualityMetrics instance
    """
    # Placeholder for actual quality metric calculations
    # In a real implementation, these would use image analysis algorithms

    return QualityMetrics(
        sharpness_score=0.85,
        contrast_score=0.90,
        brightness_score=0.75,
        text_clarity_score=0.88,
        noise_level=0.15,
        skew_angle=0.5,
        overall_quality=0.84,
    )
