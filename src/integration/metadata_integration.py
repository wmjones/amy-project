"""
Integration module for connecting OCR + AI pipeline with existing metadata systems.
Provides seamless data flow between new AI-driven processing and legacy metadata infrastructure.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime
import re

from src.metadata_extraction.extractor import DocumentMetadata, DateInfo, Entity
from src.metadata_extraction.storage import MetadataStorage
from src.metadata_extraction.ai_summarizer import DocumentSummary
from src.metadata_extraction.ocr_ai_pipeline import PipelineResult
from src.organization_logic.engine import OrganizationEngine

logger = logging.getLogger(__name__)


@dataclass
class MetadataConflict:
    """Represents a conflict between different metadata sources."""
    field_name: str
    existing_value: Any
    new_value: Any
    source: str
    resolution: Optional[str] = None
    confidence_existing: float = 0.0
    confidence_new: float = 0.0


class MetadataIntegrationBridge:
    """Bridge between AI-driven metadata and existing metadata systems."""
    
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        organization_engine: OrganizationEngine,
        conflict_resolution_strategy: str = "confidence_based"
    ):
        """Initialize the integration bridge.
        
        Args:
            metadata_storage: Existing metadata storage system
            organization_engine: Existing organization logic engine
            conflict_resolution_strategy: Strategy for resolving conflicts
                - "confidence_based": Use highest confidence value
                - "ai_preferred": Prefer AI-generated metadata
                - "existing_preferred": Prefer existing metadata
                - "manual": Flag for manual review
        """
        self.metadata_storage = metadata_storage
        self.organization_engine = organization_engine
        self.conflict_resolution_strategy = conflict_resolution_strategy
        
        # Initialize mapping tables
        self._init_category_mappings()
        self._init_entity_mappings()
        
        # Audit trail
        self.audit_trail = []
        
        logger.info(f"Metadata integration bridge initialized with {conflict_resolution_strategy} strategy")
    
    def _init_category_mappings(self):
        """Initialize category mappings between AI and existing systems."""
        self.category_mappings = {
            # AI categories -> Existing categories
            "photo": ["personal", "historical"],
            "letter": ["correspondence", "personal"],
            "business_record": ["financial", "work"],
            "certificate": ["legal", "personal"],
            "newspaper": ["reference", "historical"],
            "invoice": ["financial"],
            "contract": ["legal"],
            "report": ["work", "reference"]
        }
        
        # Reverse mappings
        self.reverse_category_mappings = {}
        for ai_cat, existing_cats in self.category_mappings.items():
            for existing_cat in existing_cats:
                if existing_cat not in self.reverse_category_mappings:
                    self.reverse_category_mappings[existing_cat] = []
                self.reverse_category_mappings[existing_cat].append(ai_cat)
    
    def _init_entity_mappings(self):
        """Initialize entity type mappings."""
        self.entity_mappings = {
            # AI entity types -> Existing entity types
            "people": "person",
            "organizations": "organization",
            "locations": "location",
            "dates": "date"
        }
    
    def integrate_pipeline_result(
        self,
        pipeline_result: PipelineResult,
        existing_metadata: Optional[DocumentMetadata] = None,
        force_update: bool = False
    ) -> Tuple[DocumentMetadata, List[MetadataConflict]]:
        """Integrate pipeline results with existing metadata system.
        
        Args:
            pipeline_result: Result from OCR + AI pipeline
            existing_metadata: Existing metadata if available
            force_update: Force update even with conflicts
            
        Returns:
            Tuple of (integrated metadata, list of conflicts)
        """
        try:
            # Extract AI-generated metadata
            ai_metadata = self._convert_ai_to_document_metadata(pipeline_result)
            
            # If no existing metadata, return AI metadata
            if not existing_metadata:
                self._log_audit_event(
                    "metadata_created",
                    pipeline_result.file_path,
                    {"source": "ai_pipeline"}
                )
                return ai_metadata, []
            
            # Merge metadata and detect conflicts
            merged_metadata, conflicts = self._merge_metadata(
                existing_metadata,
                ai_metadata,
                pipeline_result.ai_summary.confidence_score
            )
            
            # Resolve conflicts based on strategy
            if conflicts and not force_update:
                resolved_metadata = self._resolve_conflicts(
                    merged_metadata,
                    conflicts,
                    existing_metadata,
                    ai_metadata
                )
            else:
                resolved_metadata = merged_metadata
            
            # Log audit trail
            self._log_audit_event(
                "metadata_integrated",
                pipeline_result.file_path,
                {
                    "conflicts": len(conflicts),
                    "strategy": self.conflict_resolution_strategy,
                    "forced": force_update
                }
            )
            
            return resolved_metadata, conflicts
            
        except Exception as e:
            logger.error(f"Error integrating pipeline result: {e}")
            raise
    
    def _convert_ai_to_document_metadata(self, pipeline_result: PipelineResult) -> DocumentMetadata:
        """Convert AI pipeline result to DocumentMetadata format.
        
        Args:
            pipeline_result: Result from pipeline
            
        Returns:
            DocumentMetadata object
        """
        ai_summary = pipeline_result.ai_summary
        
        # Map categories
        mapped_categories = []
        if ai_summary.category in self.category_mappings:
            mapped_categories = self.category_mappings[ai_summary.category]
        else:
            mapped_categories = [ai_summary.category]
        
        # Convert entities
        entities = []
        for entity_type, names in ai_summary.key_entities.items():
            mapped_type = self.entity_mappings.get(entity_type, entity_type)
            for name in names:
                entities.append(Entity(
                    name=name,
                    type=mapped_type,
                    confidence=ai_summary.confidence_score
                ))
        
        # Convert dates
        date_info = DateInfo()
        if ai_summary.date_references:
            date_info.document_date = self._normalize_date(ai_summary.date_references[0])
            date_info.mentioned_dates = [
                self._normalize_date(d) for d in ai_summary.date_references[1:]
                if self._normalize_date(d)
            ]
        
        # Create DocumentMetadata
        return DocumentMetadata(
            document_type=self._map_content_type_to_document_type(ai_summary.content_type),
            categories=mapped_categories,
            dates=date_info,
            entities=entities,
            topics=ai_summary.photo_subjects + ai_summary.classification_tags,
            tags=ai_summary.classification_tags,
            summary=ai_summary.summary,
            suggested_folder=ai_summary.suggested_folder_path,
            confidence_score=ai_summary.confidence_score,
            source_file=str(pipeline_result.file_path),
            processing_timestamp=datetime.now().isoformat(),
            extracted_text=pipeline_result.ocr_result.text[:1000],
            language="en"
        )
    
    def _map_content_type_to_document_type(self, content_type: str) -> str:
        """Map AI content type to existing document type."""
        mappings = {
            "photo": "image",
            "document": "text",
            "mixed": "composite",
            "unknown": "general"
        }
        return mappings.get(content_type, content_type)
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to ISO-8601 format."""
        if not date_str:
            return None
        
        # Check if already ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Try to extract year for partial dates
        year_match = re.search(r'\b(18\d{2}|19\d{2}|20\d{2})\b', date_str)
        if year_match:
            return f"{year_match.group(0)}-01-01"
        
        return None
    
    def _merge_metadata(
        self,
        existing: DocumentMetadata,
        ai_generated: DocumentMetadata,
        ai_confidence: float
    ) -> Tuple[DocumentMetadata, List[MetadataConflict]]:
        """Merge existing and AI-generated metadata.
        
        Args:
            existing: Existing metadata
            ai_generated: AI-generated metadata
            ai_confidence: Confidence score from AI
            
        Returns:
            Tuple of (merged metadata, conflicts)
        """
        conflicts = []
        merged = DocumentMetadata(
            document_type=existing.document_type,
            categories=existing.categories.copy(),
            dates=existing.dates,
            entities=existing.entities.copy(),
            topics=existing.topics.copy(),
            tags=existing.tags.copy(),
            summary=existing.summary,
            suggested_folder=existing.suggested_folder,
            confidence_score=existing.confidence_score,
            source_file=existing.source_file,
            processing_timestamp=datetime.now().isoformat(),
            extracted_text=existing.extracted_text,
            file_size=existing.file_size,
            page_count=existing.page_count,
            language=existing.language
        )
        
        # Check document type
        if existing.document_type != ai_generated.document_type:
            conflicts.append(MetadataConflict(
                field_name="document_type",
                existing_value=existing.document_type,
                new_value=ai_generated.document_type,
                source="ai_pipeline",
                confidence_existing=existing.confidence_score,
                confidence_new=ai_confidence
            ))
        
        # Merge categories
        new_categories = set(ai_generated.categories) - set(existing.categories)
        if new_categories:
            merged.categories.extend(list(new_categories))
        
        # Check dates
        if ai_generated.dates.document_date and existing.dates.document_date:
            if ai_generated.dates.document_date != existing.dates.document_date:
                conflicts.append(MetadataConflict(
                    field_name="document_date",
                    existing_value=existing.dates.document_date,
                    new_value=ai_generated.dates.document_date,
                    source="ai_pipeline",
                    confidence_existing=existing.confidence_score,
                    confidence_new=ai_confidence
                ))
        elif ai_generated.dates.document_date and not existing.dates.document_date:
            merged.dates.document_date = ai_generated.dates.document_date
        
        # Merge entities
        existing_entities = {(e.name, e.type) for e in existing.entities}
        for entity in ai_generated.entities:
            if (entity.name, entity.type) not in existing_entities:
                merged.entities.append(entity)
        
        # Merge topics and tags
        merged.topics = list(set(merged.topics + ai_generated.topics))
        merged.tags = list(set(merged.tags + ai_generated.tags))
        
        # Handle summary
        if ai_generated.summary and (not existing.summary or len(ai_generated.summary) > len(existing.summary)):
            merged.summary = ai_generated.summary
        
        # Handle suggested folder
        if ai_confidence > existing.confidence_score and ai_generated.suggested_folder:
            merged.suggested_folder = ai_generated.suggested_folder
        
        # Update confidence score
        merged.confidence_score = max(existing.confidence_score, ai_confidence)
        
        # Update extracted text if AI version is longer
        if ai_generated.extracted_text and len(ai_generated.extracted_text) > len(existing.extracted_text or ""):
            merged.extracted_text = ai_generated.extracted_text
        
        return merged, conflicts
    
    def _resolve_conflicts(
        self,
        merged_metadata: DocumentMetadata,
        conflicts: List[MetadataConflict],
        existing: DocumentMetadata,
        ai_generated: DocumentMetadata
    ) -> DocumentMetadata:
        """Resolve metadata conflicts based on strategy.
        
        Args:
            merged_metadata: Partially merged metadata
            conflicts: List of conflicts
            existing: Original existing metadata
            ai_generated: AI-generated metadata
            
        Returns:
            Resolved metadata
        """
        for conflict in conflicts:
            resolution = None
            
            if self.conflict_resolution_strategy == "confidence_based":
                if conflict.confidence_new > conflict.confidence_existing:
                    resolution = "use_new"
                else:
                    resolution = "keep_existing"
            
            elif self.conflict_resolution_strategy == "ai_preferred":
                resolution = "use_new"
            
            elif self.conflict_resolution_strategy == "existing_preferred":
                resolution = "keep_existing"
            
            elif self.conflict_resolution_strategy == "manual":
                resolution = "manual_review"
                conflict.resolution = "manual_review_required"
            
            # Apply resolution
            if resolution == "use_new":
                if conflict.field_name == "document_type":
                    merged_metadata.document_type = conflict.new_value
                elif conflict.field_name == "document_date":
                    merged_metadata.dates.document_date = conflict.new_value
                conflict.resolution = "accepted_new"
            
            elif resolution == "keep_existing":
                conflict.resolution = "kept_existing"
        
        return merged_metadata
    
    def update_organization_path(
        self,
        metadata: DocumentMetadata,
        use_ai_suggestion: bool = True
    ) -> Tuple[str, str]:
        """Update organization path using integrated metadata.
        
        Args:
            metadata: Integrated metadata
            use_ai_suggestion: Whether to consider AI suggestions
            
        Returns:
            Tuple of (target_path, rule_name)
        """
        # Let the organization engine determine the path
        target_path, rule_name = self.organization_engine.determine_target_location(metadata)
        
        # If using AI suggestions and confidence is high, consider AI path
        if use_ai_suggestion and metadata.suggested_folder and metadata.confidence_score >= 0.85:
            # Check if AI suggestion is more specific
            ai_path_parts = metadata.suggested_folder.split('/')
            engine_path_parts = target_path.split('/')
            
            if len(ai_path_parts) > len(engine_path_parts):
                target_path = metadata.suggested_folder
                rule_name = "AI Enhanced Organization"
        
        return target_path, rule_name
    
    def create_enriched_metadata(
        self,
        pipeline_results: List[PipelineResult],
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create enriched metadata report for a batch of documents.
        
        Args:
            pipeline_results: List of pipeline results
            batch_name: Optional name for the batch
            
        Returns:
            Enriched metadata report
        """
        report = {
            "batch_name": batch_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processing_date": datetime.now().isoformat(),
            "total_documents": len(pipeline_results),
            "enrichment_statistics": {
                "ocr_success": 0,
                "ai_classification_success": 0,
                "metadata_conflicts": 0,
                "new_entities_discovered": 0,
                "date_references_found": 0,
                "location_references_found": 0
            },
            "document_distribution": {},
            "syracuse_specific": {
                "historical_periods": {},
                "locations": {},
                "related_themes": []
            },
            "integration_summary": {
                "new_metadata_created": 0,
                "existing_metadata_updated": 0,
                "conflicts_resolved": 0,
                "manual_review_required": 0
            },
            "documents": []
        }
        
        # Process each result
        for result in pipeline_results:
            if result.success:
                report["enrichment_statistics"]["ocr_success"] += 1
                
                if result.ai_summary.category != "error":
                    report["enrichment_statistics"]["ai_classification_success"] += 1
                
                # Count entities and references
                entity_count = sum(len(v) for v in result.ai_summary.key_entities.values())
                report["enrichment_statistics"]["new_entities_discovered"] += entity_count
                
                report["enrichment_statistics"]["date_references_found"] += len(
                    result.ai_summary.date_references
                )
                
                report["enrichment_statistics"]["location_references_found"] += len(
                    result.ai_summary.location_references
                )
                
                # Document distribution
                category = result.ai_summary.category
                report["document_distribution"][category] = \
                    report["document_distribution"].get(category, 0) + 1
                
                # Syracuse-specific data
                period = result.ai_summary.historical_period
                if period:
                    report["syracuse_specific"]["historical_periods"][period] = \
                        report["syracuse_specific"]["historical_periods"].get(period, 0) + 1
                
                for location in result.ai_summary.location_references:
                    report["syracuse_specific"]["locations"][location] = \
                        report["syracuse_specific"]["locations"].get(location, 0) + 1
                
                # Document summary
                doc_summary = {
                    "file_path": str(result.file_path),
                    "category": result.ai_summary.category,
                    "confidence": result.ai_summary.confidence_score,
                    "summary": result.ai_summary.summary[:200] + "...",
                    "suggested_path": result.ai_summary.suggested_folder_path,
                    "key_entities": result.ai_summary.key_entities,
                    "processing_time": result.processing_time
                }
                
                report["documents"].append(doc_summary)
        
        # Extract themes
        all_tags = []
        for result in pipeline_results:
            if result.success:
                all_tags.extend(result.ai_summary.classification_tags)
        
        # Count tag frequency for themes
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Top themes
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        report["syracuse_specific"]["related_themes"] = [
            tag for tag, count in sorted_tags[:10]
        ]
        
        return report
    
    def validate_integration(
        self,
        integrated_metadata: DocumentMetadata,
        original_metadata: Optional[DocumentMetadata] = None
    ) -> Dict[str, Any]:
        """Validate integrated metadata for consistency.
        
        Args:
            integrated_metadata: Integrated metadata
            original_metadata: Original metadata if available
            
        Returns:
            Validation report
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "improvements": []
        }
        
        # Check required fields
        if not integrated_metadata.document_type:
            validation["errors"].append("Missing document type")
            validation["is_valid"] = False
        
        if not integrated_metadata.categories:
            validation["errors"].append("Missing categories")
            validation["is_valid"] = False
        
        if integrated_metadata.confidence_score < 0 or integrated_metadata.confidence_score > 1:
            validation["errors"].append("Invalid confidence score")
            validation["is_valid"] = False
        
        # Check path safety
        if integrated_metadata.suggested_folder:
            invalid_chars = '<>:"|?*'
            if any(char in integrated_metadata.suggested_folder for char in invalid_chars):
                validation["warnings"].append("Suggested folder contains invalid characters")
        
        # Compare with original if available
        if original_metadata:
            # Check for improvements
            if len(integrated_metadata.entities) > len(original_metadata.entities):
                validation["improvements"].append(
                    f"Added {len(integrated_metadata.entities) - len(original_metadata.entities)} new entities"
                )
            
            if integrated_metadata.summary and not original_metadata.summary:
                validation["improvements"].append("Added document summary")
            
            if integrated_metadata.dates.document_date and not original_metadata.dates.document_date:
                validation["improvements"].append("Added document date")
            
            if integrated_metadata.confidence_score > original_metadata.confidence_score:
                validation["improvements"].append(
                    f"Improved confidence from {original_metadata.confidence_score:.2f} to {integrated_metadata.confidence_score:.2f}"
                )
        
        # Check consistency
        if integrated_metadata.dates.document_date:
            try:
                datetime.fromisoformat(integrated_metadata.dates.document_date)
            except ValueError:
                validation["warnings"].append("Invalid date format")
        
        return validation
    
    def _log_audit_event(self, event_type: str, file_path: str, details: Dict[str, Any]):
        """Log an audit trail event.
        
        Args:
            event_type: Type of event
            file_path: File being processed
            details: Event details
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "file_path": file_path,
            "details": details
        }
        
        self.audit_trail.append(audit_entry)
        logger.info(f"Audit: {event_type} for {file_path}")
    
    def get_audit_trail(
        self,
        file_path: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered audit trail.
        
        Args:
            file_path: Filter by file path
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Filtered audit trail
        """
        filtered_trail = self.audit_trail
        
        if file_path:
            filtered_trail = [e for e in filtered_trail if e["file_path"] == file_path]
        
        if event_type:
            filtered_trail = [e for e in filtered_trail if e["event_type"] == event_type]
        
        if start_date:
            filtered_trail = [
                e for e in filtered_trail
                if datetime.fromisoformat(e["timestamp"]) >= start_date
            ]
        
        if end_date:
            filtered_trail = [
                e for e in filtered_trail
                if datetime.fromisoformat(e["timestamp"]) <= end_date
            ]
        
        return filtered_trail
    
    def export_audit_trail(self, output_path: Path):
        """Export audit trail to file.
        
        Args:
            output_path: Path for output file
        """
        with open(output_path, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        
        logger.info(f"Audit trail exported to {output_path}")