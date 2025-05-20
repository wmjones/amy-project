"""
Claude API client for document analysis - FIXED VERSION with image support.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import anthropic
from anthropic import Anthropic, APIError, RateLimitError
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from Claude document analysis."""

    content: str
    metadata: Dict[str, Any]
    confidence_score: float
    tokens_used: int
    model: str


class ClaudeClient:
    """Client for interacting with Claude API for document analysis."""

    DEFAULT_SYSTEM_PROMPT = """You are an expert document analyzer specializing in organizing files.
    Extract key information from documents including:
    - Document type and category
    - Main topics and subjects
    - Date information if available
    - Key entities (people, organizations, locations)
    - Suggested folder structure for organization

    Provide structured output that can be used to organize files intelligently."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to env variable)
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            retry_attempts: Number of retry attempts
            retry_delay: Initial retry delay in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        logger.info(f"Initialized Claude client with model: {model}")

    def analyze_document(
        self,
        content: str,
        file_name: str = "",
        file_type: str = "",
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
    ) -> AnalysisResult:
        """Analyze a document using Claude.

        Args:
            content: Document content to analyze
            file_name: Original file name
            file_type: File type/extension
            custom_prompt: Custom prompt to use
            system_prompt: Custom system prompt (defaults to class default)
            image_path: Optional path to image file to include with the analysis

        Returns:
            AnalysisResult with analysis details
        """
        # Prepare the prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._create_analysis_prompt(content, file_name, file_type)

        # Use custom or default system prompt
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = self._make_api_call(prompt, system, image_path)
                return self._parse_response(response, file_name)

            except RateLimitError as e:
                # Handle rate limiting specifically
                wait_time = self._calculate_wait_time(attempt, e)
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}"
                )
                time.sleep(wait_time)
                last_error = e

            except APIError as e:
                # Handle other API errors
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"API error: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    last_error = e
                else:
                    raise

            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error analyzing document: {e}")
                raise

        # If we exhausted retries, raise the last error
        raise last_error

    def analyze_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 5,
        progress_callback=None,
    ) -> List[AnalysisResult]:
        """Analyze multiple documents in batches.

        Args:
            documents: List of documents with 'content', 'file_name', 'file_type', and optional 'image_path'
            batch_size: Number of documents to process concurrently
            progress_callback: Optional callback for progress updates

        Returns:
            List of AnalysisResult objects
        """
        results = []
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            batch_results = []

            for doc in batch:
                try:
                    result = self.analyze_document(
                        content=doc["content"],
                        file_name=doc.get("file_name", ""),
                        file_type=doc.get("file_type", ""),
                        image_path=doc.get("image_path"),
                    )
                    batch_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error analyzing {doc.get('file_name', 'unknown')}: {e}"
                    )
                    # Create error result
                    error_result = AnalysisResult(
                        content=str(e),
                        metadata={"error": True, "error_message": str(e)},
                        confidence_score=0.0,
                        tokens_used=0,
                        model=self.model,
                    )
                    batch_results.append(error_result)

            results.extend(batch_results)

            # Update progress
            if progress_callback:
                progress_callback(len(results), total_docs)

            # Rate limit protection between batches
            if i + batch_size < total_docs:
                time.sleep(1)  # Small delay between batches

        return results

    def _encode_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Encode an image file as base64 for the API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image data for the API
        """
        path = Path(image_path)
        
        # Determine media type based on file extension
        extension = path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        media_type = media_types.get(extension, 'image/jpeg')
        
        # Read and encode the image
        with open(path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        }

    def _make_api_call(self, prompt: str, system: str, image_path: Optional[Union[str, Path]] = None) -> Any:
        """Make the actual API call to Claude.

        Args:
            prompt: User prompt
            system: System prompt
            image_path: Optional path to image file to include

        Returns:
            API response
        """
        # Build the content array
        content = []
        
        # Add image if provided
        if image_path:
            try:
                image_data = self._encode_image(image_path)
                content.append(image_data)
            except Exception as e:
                logger.warning(f"Failed to encode image {image_path}: {e}")
                # Continue without the image
        
        # Add text content
        content.append({"type": "text", "text": prompt})
        
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

    def _create_analysis_prompt(
        self, content: str, file_name: str, file_type: str
    ) -> str:
        """Create analysis prompt for document.

        Args:
            content: Document content
            file_name: File name
            file_type: File type

        Returns:
            Formatted prompt
        """
        prompt = f"""Analyze the following document and provide structured information for file organization:

File Name: {file_name}
File Type: {file_type}

Document Content:
{content[:10000]}  # Limit content for large files

Please provide:
1. Document category (e.g., financial, personal, work, medical, legal)
2. Main topics or subjects
3. Date information (if available)
4. Key entities (people, organizations, locations)
5. Suggested folder path for organization
6. Confidence score (0-1) for your analysis
7. Brief summary of the document

Format the response as structured JSON."""

        return prompt

    def _parse_response(self, response: Any, file_name: str) -> AnalysisResult:
        """Parse Claude's response into AnalysisResult.

        Args:
            response: API response
            file_name: Original file name

        Returns:
            Parsed AnalysisResult
        """
        try:
            # Extract text content from response
            content = response.content[0].text

            # Try to parse as JSON for structured data
            import json

            try:
                metadata = json.loads(content)
                confidence_score = metadata.get("confidence_score", 0.8)
            except json.JSONDecodeError:
                # If not JSON, create basic metadata
                metadata = {"raw_response": content, "file_name": file_name}
                confidence_score = 0.7

            # Calculate token usage - FIXED: use input_tokens + output_tokens
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return AnalysisResult(
                content=content,
                metadata=metadata,
                confidence_score=confidence_score,
                tokens_used=tokens_used,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise

    def _calculate_wait_time(self, attempt: int, error: RateLimitError) -> float:
        """Calculate wait time for rate limit errors.

        Args:
            attempt: Current attempt number
            error: Rate limit error

        Returns:
            Wait time in seconds
        """
        # Check if retry-after header is available
        if hasattr(error, "response") and error.response:
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # Default exponential backoff
        base_wait = self.retry_delay * (2**attempt)
        jitter = base_wait * 0.1  # Add 10% jitter
        return base_wait + jitter