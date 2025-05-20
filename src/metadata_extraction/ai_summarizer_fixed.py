"""
AI-driven text summarization and classification module for the Hansman Syracuse collection.
Integrates with Claude API for intelligent document processing.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import re

from src.claude_integration.client import ClaudeClient, AnalysisResult
from src.claude_integration.prompts import PromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class DocumentSummary:
    """Enhanced summary for historical documents in the Hansman Syracuse collection."""

    file_path: str
    ocr_text: str
    summary: str
    category: str
    confidence_score: float
    key_entities: Dict[str, List[str]]
    date_references: List[str]
    photo_subjects: List[str]
    location_references: List[str]
    content_type: str  # photo, document, mixed
    historical_period: str
    classification_tags: List[str]
    claude_metadata: Dict[str, Any]
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    suggested_folder_path: str = ""
    related_documents: List[str] = field(default_factory=list)
    quality_indicators: Dict[str, float] = field(default_factory=dict)


class AISummarizer:
    """AI-driven summarization and classification for historical documents."""

    HANSMAN_SPECIFIC_PROMPT = """You are analyzing a historical document from the Hansman Syracuse photo collection from July 2015. This collection contains:
- Historical photographs of Syracuse, NY
- Documents related to Syracuse history
- Family photos and documents
- Business records and correspondence
- Architectural and landscape photos

Please analyze the following text extracted via OCR and provide structured information. Note that the original image is included for visual context. Return ONLY valid JSON without any additional text before or after:

{{
    "summary": "Comprehensive summary of the document/photo content",
    "category": "Type of document (photo, letter, business_record, newspaper, certificate, etc.)",
    "confidence_score": 0.0-1.0,
    "key_entities": {{
        "people": ["Names of people mentioned or shown"],
        "organizations": ["Businesses, institutions, etc."],
        "locations": ["Specific Syracuse locations or other places"],
        "dates": ["Any date references found"]
    }},
    "photo_subjects": ["If photo: main subjects visible"],
    "historical_period": "Estimated time period (e.g., '1920s', 'early 1900s')",
    "content_type": "photo/document/mixed",
    "location_references": ["Syracuse neighborhoods, streets, landmarks"],
    "classification_tags": ["searchable tags for cataloging"],
    "suggested_folder_path": "Hansman_Syracuse/[category]/[time_period]/[specific_topic]",
    "quality_indicators": {{
        "text_clarity": 0.0-1.0,
        "historical_value": 0.0-1.0,
        "preservation_priority": 0.0-1.0
    }},
    "related_themes": ["Architecture", "Local Business", "Family History", etc.]
}}

OCR Text:
{ocr_text}

If the text is unclear or minimal, use both the OCR text and the attached image to make educated inferences based on visible content and typical patterns in historical Syracuse documents."""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        """Initialize the AI summarizer.

        Args:
            claude_client: Optional pre-configured Claude client
        """
        self.claude_client = claude_client or ClaudeClient()
        self.prompt_templates = PromptTemplates()
        self._init_syracuse_knowledge_base()

    def _init_syracuse_knowledge_base(self):
        """Initialize knowledge base specific to Syracuse history."""
        self.syracuse_landmarks = [
            "Armory Square",
            "Clinton Square",
            "Hanover Square",
            "University Hill",
            "Eastwood",
            "Strathmore",
            "Salt Springs",
            "Onondaga Lake",
            "Erie Canal",
        ]

        self.historical_periods = {
            "salt_era": (1800, 1920),
            "canal_era": (1825, 1950),
            "industrial_boom": (1870, 1930),
            "urban_renewal": (1950, 1980),
            "modern": (1980, 2015),
        }

        self.document_patterns = {
            "photo": ["photograph", "picture", "image", "snapshot"],
            "letter": ["dear", "sincerely", "regards", "wrote"],
            "business": ["invoice", "receipt", "company", "incorporated"],
            "newspaper": ["times", "herald", "journal", "post-standard"],
            "certificate": ["certificate", "certify", "awarded", "granted"],
        }

    def summarize_document(
        self,
        file_path: Path,
        ocr_text: str,
        ocr_confidence: float = 0.0,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> DocumentSummary:
        """Generate AI-driven summary and classification for a document.

        Args:
            file_path: Path to the document
            ocr_text: Extracted text from OCR
            ocr_confidence: OCR confidence score
            additional_context: Additional metadata or context

        Returns:
            DocumentSummary with analysis results
        """
        import time

        start_time = time.time()

        try:
            # Prepare enhanced prompt with Syracuse-specific context
            prompt = self._prepare_syracuse_prompt(
                ocr_text, file_path, additional_context
            )

            # Get analysis from Claude - now with image attachment support
            # Check if the file is an image
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            include_image = file_path.suffix.lower() in image_extensions

            result = self.claude_client.analyze_document(
                content=ocr_text,
                file_name=str(file_path.name),
                file_type=file_path.suffix,
                custom_prompt=prompt,
                system_prompt=self.HANSMAN_SPECIFIC_PROMPT,
                image_path=file_path if include_image else None,
            )

            # Parse the response
            summary_data = self._parse_claude_response(result)

            # Enhance with local knowledge
            enhanced_data = self._enhance_with_local_knowledge(summary_data, ocr_text)

            # Create document summary
            processing_time = time.time() - start_time

            document_summary = DocumentSummary(
                file_path=str(file_path),
                ocr_text=ocr_text,
                summary=enhanced_data.get("summary", ""),
                category=enhanced_data.get("category", "unknown"),
                confidence_score=max(
                    enhanced_data.get("confidence_score", 0.5), ocr_confidence
                ),
                key_entities=enhanced_data.get("key_entities", {}),
                date_references=enhanced_data.get("date_references", []),
                photo_subjects=enhanced_data.get("photo_subjects", []),
                location_references=enhanced_data.get("location_references", []),
                content_type=enhanced_data.get("content_type", "unknown"),
                historical_period=enhanced_data.get("historical_period", "unknown"),
                classification_tags=enhanced_data.get("classification_tags", []),
                claude_metadata={
                    "model": result.model,
                    "tokens_used": result.tokens_used,
                    "raw_response": result.content,
                },
                processing_time=processing_time,
                suggested_folder_path=enhanced_data.get("suggested_folder_path", ""),
                quality_indicators=enhanced_data.get("quality_indicators", {}),
            )

            return document_summary

        except Exception as e:
            logger.error(f"Error summarizing document {file_path}: {e}")
            return DocumentSummary(
                file_path=str(file_path),
                ocr_text=ocr_text,
                summary="Error during summarization",
                category="error",
                confidence_score=0.0,
                key_entities={},
                date_references=[],
                photo_subjects=[],
                location_references=[],
                content_type="unknown",
                historical_period="unknown",
                classification_tags=[],
                claude_metadata={},
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def _prepare_syracuse_prompt(
        self,
        ocr_text: str,
        file_path: Path,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Prepare a Syracuse-specific prompt for Claude."""
        context_parts = []

        # Add filename analysis
        filename_info = self._analyze_filename(file_path.name)
        if filename_info:
            context_parts.append(f"Filename analysis: {filename_info}")

        # Add any additional context
        if additional_context:
            context_parts.append(
                f"Additional context: {json.dumps(additional_context)}"
            )

        # Check for Syracuse references
        syracuse_refs = self._find_syracuse_references(ocr_text)
        if syracuse_refs:
            context_parts.append(
                f"Syracuse references found: {', '.join(syracuse_refs)}"
            )

        prompt = self.HANSMAN_SPECIFIC_PROMPT.format(ocr_text=ocr_text)

        if context_parts:
            prompt += f"\n\nAdditional Context:\n" + "\n".join(context_parts)

        return prompt

    def _parse_claude_response(self, result: AnalysisResult) -> Dict[str, Any]:
        """Parse Claude's response into structured data. Enhanced with better error handling."""
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw Claude response: {result.content[:500]}")

            # Clean the response content - remove leading/trailing whitespace and newlines
            content = result.content.strip()

            # Handle responses that start with markdown code blocks
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()

            # Try direct JSON parsing first
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parse failed: {e}")

            # Extract JSON using regex - more robust pattern
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            json_matches = re.findall(json_pattern, content, re.DOTALL)

            if json_matches:
                # Try the largest JSON match first (likely to be the most complete)
                json_matches.sort(key=len, reverse=True)

                for json_str in json_matches:
                    try:
                        # Clean up common issues
                        json_str = json_str.strip()
                        # Fix single quotes
                        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                        # Fix trailing commas
                        json_str = re.sub(r",\s*}", "}", json_str)
                        json_str = re.sub(r",\s*]", "]", json_str)
                        # Fix unquoted keys
                        json_str = re.sub(r"(\w+):", r'"\1":', json_str)

                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON match: {e}")
                        continue

            # Try to find JSON between first { and last }
            json_start = content.find("{")
            json_end = content.rfind("}")

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = content[json_start : json_end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try with fixes
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r",\s*}", "}", json_str)
                    json_str = re.sub(r",\s*]", "]", json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse extracted JSON: {e}")

            # If all JSON parsing fails, extract structure from text
            logger.warning("No valid JSON found, attempting regex extraction")
            return self._extract_structure_from_text(content)

        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.debug(f"Failed content: {result.content[:500]}")
            return self._extract_structure_from_text(result.content)

    def _extract_structure_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from unstructured text response using regex patterns."""
        structure = {
            "summary": "",
            "category": "unknown",
            "confidence_score": 0.7,
            "key_entities": {
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
            },
            "date_references": [],
            "photo_subjects": [],
            "location_references": [],
            "content_type": "unknown",
            "historical_period": "unknown",
            "classification_tags": [],
            "suggested_folder_path": "Hansman_Syracuse/Uncategorized",
            "quality_indicators": {
                "text_clarity": 0.5,
                "historical_value": 0.5,
                "preservation_priority": 0.5,
            },
        }

        # Try to extract summary
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', text)
        if summary_match:
            structure["summary"] = summary_match.group(1)
        else:
            # Use first 500 chars as fallback
            structure["summary"] = text[:500].strip()

        # Extract category
        category_match = re.search(r'"category"\s*:\s*"([^"]*)"', text)
        if category_match:
            structure["category"] = category_match.group(1)

        # Extract dates
        date_pattern = r"\b(18\d{2}|19\d{2}|20\d{2})\b"
        dates = re.findall(date_pattern, text)
        structure["date_references"] = list(set(dates))
        structure["key_entities"]["dates"] = structure["date_references"]

        # Extract confidence score
        confidence_match = re.search(r'"confidence_score"\s*:\s*([0-9.]+)', text)
        if confidence_match:
            try:
                structure["confidence_score"] = float(confidence_match.group(1))
            except ValueError:
                pass

        # Extract potential locations
        for landmark in self.syracuse_landmarks:
            if landmark.lower() in text.lower():
                structure["location_references"].append(landmark)
                structure["key_entities"]["locations"].append(landmark)

        # Extract content type
        content_type_match = re.search(r'"content_type"\s*:\s*"([^"]*)"', text)
        if content_type_match:
            structure["content_type"] = content_type_match.group(1)
        else:
            # Guess content type
            text_lower = text.lower()
            for doc_type, patterns in self.document_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    structure["content_type"] = doc_type
                    structure["category"] = doc_type
                    break

        # Extract historical period
        period_match = re.search(r'"historical_period"\s*:\s*"([^"]*)"', text)
        if period_match:
            structure["historical_period"] = period_match.group(1)

        # Extract suggested folder path
        path_match = re.search(r'"suggested_folder_path"\s*:\s*"([^"]*)"', text)
        if path_match:
            structure["suggested_folder_path"] = path_match.group(1)

        return structure

    def _enhance_with_local_knowledge(
        self, data: Dict[str, Any], ocr_text: str
    ) -> Dict[str, Any]:
        """Enhance Claude's analysis with local Syracuse knowledge."""
        # Add Syracuse-specific location recognition
        text_lower = ocr_text.lower()

        for landmark in self.syracuse_landmarks:
            if landmark.lower() in text_lower and landmark not in data.get(
                "location_references", []
            ):
                data.setdefault("location_references", []).append(landmark)

        # Determine historical period based on dates
        if data.get("date_references"):
            try:
                years = [
                    int(d)
                    for d in data["date_references"]
                    if d.isdigit() and len(d) == 4
                ]
                if years:
                    avg_year = sum(years) / len(years)
                    for period_name, (start, end) in self.historical_periods.items():
                        if start <= avg_year <= end:
                            data["historical_period"] = period_name
                            break
            except ValueError:
                pass

        # Add specific tags for Syracuse content
        if "syracuse" in text_lower:
            data.setdefault("classification_tags", []).append("syracuse_local")

        if "salt" in text_lower and "springs" in text_lower:
            data.setdefault("classification_tags", []).append("salt_industry")

        if "erie canal" in text_lower:
            data.setdefault("classification_tags", []).append("erie_canal")

        # Enhance folder path suggestion
        if (
            not data.get("suggested_folder_path")
            or data["suggested_folder_path"] == "Hansman_Syracuse/Uncategorized"
        ):
            category = data.get("category", "unknown")
            period = data.get("historical_period", "unknown")
            folder_path = f"Hansman_Syracuse/{category}/{period}"

            if data.get("location_references"):
                folder_path += f"/{data['location_references'][0].replace(' ', '_')}"

            data["suggested_folder_path"] = folder_path

        return data

    def _analyze_filename(self, filename: str) -> str:
        """Analyze filename for clues about content."""
        patterns = {
            r"(\d{4})": "possible year",
            r"syracuse": "Syracuse reference",
            r"photo|img|pic": "likely photograph",
            r"doc|letter|cert": "likely document",
            r"(\d+)_(\d+)": "possible date or ID",
        }

        findings = []
        for pattern, description in patterns.items():
            if re.search(pattern, filename, re.IGNORECASE):
                findings.append(description)

        return ", ".join(findings) if findings else ""

    def _find_syracuse_references(self, text: str) -> List[str]:
        """Find Syracuse-specific references in text."""
        references = []
        text_lower = text.lower()

        # Check for landmarks
        for landmark in self.syracuse_landmarks:
            if landmark.lower() in text_lower:
                references.append(landmark)

        # Check for common Syracuse terms
        syracuse_terms = [
            "syracuse",
            "onondaga",
            "salt city",
            "central new york",
            "carrier dome",
            "syracuse university",
            "destiny usa",
        ]

        for term in syracuse_terms:
            if term in text_lower:
                references.append(term.title())

        return list(set(references))

    def batch_summarize(
        self,
        documents: List[Tuple[Path, str, Optional[Dict[str, Any]]]],
        batch_size: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> List[DocumentSummary]:
        """Batch process multiple documents for summarization.

        Args:
            documents: List of (file_path, ocr_text, optional_context) tuples
            batch_size: Number of documents to process concurrently
            progress_callback: Optional callback for progress updates

        Returns:
            List of DocumentSummary objects
        """
        summaries = []
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_summaries = []

            for file_path, ocr_text, context in batch:
                summary = self.summarize_document(
                    file_path=file_path, ocr_text=ocr_text, additional_context=context
                )
                batch_summaries.append(summary)

            summaries.extend(batch_summaries)

            if progress_callback:
                progress_callback(len(summaries), total)

        return summaries

    def create_collection_report(
        self, summaries: List[DocumentSummary], output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create a comprehensive report of the collection analysis.

        Args:
            summaries: List of document summaries
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the analysis report
        """
        report = {
            "collection": "Hansman Syracuse Photo Documents",
            "processing_date": datetime.now().isoformat(),
            "total_documents": len(summaries),
            "statistics": {
                "by_category": {},
                "by_period": {},
                "locations_mentioned": {},
                "confidence_distribution": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.5-0.8
                    "low": 0,  # < 0.5
                },
            },
            "key_findings": [],
            "error_count": 0,
            "processing_time": sum(s.processing_time for s in summaries),
        }

        # Analyze summaries
        for summary in summaries:
            # Count by category
            category = summary.category
            report["statistics"]["by_category"][category] = (
                report["statistics"]["by_category"].get(category, 0) + 1
            )

            # Count by period
            period = summary.historical_period
            report["statistics"]["by_period"][period] = (
                report["statistics"]["by_period"].get(period, 0) + 1
            )

            # Count locations
            for location in summary.location_references:
                report["statistics"]["locations_mentioned"][location] = (
                    report["statistics"]["locations_mentioned"].get(location, 0) + 1
                )

            # Confidence distribution
            if summary.confidence_score > 0.8:
                report["statistics"]["confidence_distribution"]["high"] += 1
            elif summary.confidence_score > 0.5:
                report["statistics"]["confidence_distribution"]["medium"] += 1
            else:
                report["statistics"]["confidence_distribution"]["low"] += 1

            # Count errors
            if summary.error_message:
                report["error_count"] += 1

        # Generate key findings
        if report["statistics"]["locations_mentioned"]:
            most_mentioned = max(
                report["statistics"]["locations_mentioned"].items(), key=lambda x: x[1]
            )
            report["key_findings"].append(
                f"Most mentioned location: {most_mentioned[0]} ({most_mentioned[1]} times)"
            )

        if report["statistics"]["by_period"]:
            dominant_period = max(
                report["statistics"]["by_period"].items(), key=lambda x: x[1]
            )
            report["key_findings"].append(
                f"Most common historical period: {dominant_period[0]}"
            )

        # Save report if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report
