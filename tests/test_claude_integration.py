"""
Unit tests for Claude API integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.claude_integration.client import ClaudeClient, AnalysisResult
from src.claude_integration.prompts import PromptTemplates
from anthropic import APIError, RateLimitError
import json


class TestClaudeClient:
    """Test ClaudeClient functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Claude client."""
        with patch("src.claude_integration.client.Anthropic") as mock_anthropic:
            # Create mock instance
            mock_instance = Mock()
            mock_anthropic.return_value = mock_instance

            # Create client with mocked Anthropic
            client = ClaudeClient(api_key="test_key")
            client.client = mock_instance

            yield client, mock_instance

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = ClaudeClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.model == "claude-3-7-sonnet-20250219"
        assert client.max_tokens == 4000
        assert client.temperature == 0.7

    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No API key provided"):
                ClaudeClient()

    def test_analyze_document_success(self, mock_client):
        """Test successful document analysis."""
        client, mock_instance = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text=json.dumps(
                    {
                        "document_type": "invoice",
                        "category": "financial",
                        "confidence_score": 0.95,
                    }
                )
            )
        ]
        mock_response.usage.total_tokens = 100

        mock_instance.messages.create.return_value = mock_response

        # Test analysis
        result = client.analyze_document(
            content="Invoice content", file_name="invoice.pdf", file_type="pdf"
        )

        assert isinstance(result, AnalysisResult)
        assert result.metadata["document_type"] == "invoice"
        assert result.confidence_score == 0.95
        assert result.tokens_used == 100

    def test_analyze_document_retry_on_rate_limit(self, mock_client):
        """Test retry logic on rate limit errors."""
        client, mock_instance = mock_client

        # Mock rate limit error then success
        mock_http_response = Mock(headers={}, status_code=429)
        rate_limit_error = RateLimitError(
            "Rate limit exceeded", response=mock_http_response, body=None
        )
        mock_response = Mock()
        mock_response.content = [Mock(text='{"result": "success"}')]
        mock_response.usage.total_tokens = 50

        mock_instance.messages.create.side_effect = [rate_limit_error, mock_response]

        # Test with mocked time.sleep
        with patch("time.sleep"):
            result = client.analyze_document("test content")

        assert result.tokens_used == 50
        assert mock_instance.messages.create.call_count == 2

    def test_analyze_document_max_retries_exceeded(self, mock_client):
        """Test max retries exceeded raises error."""
        client, mock_instance = mock_client
        client.retry_attempts = 2

        # Mock continuous rate limit errors
        mock_http_response = Mock(headers={}, status_code=429)
        rate_limit_error = RateLimitError(
            "Rate limit exceeded", response=mock_http_response, body=None
        )
        mock_instance.messages.create.side_effect = rate_limit_error

        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                client.analyze_document("test content")

        assert mock_instance.messages.create.call_count == 2

    def test_analyze_batch(self, mock_client):
        """Test batch document analysis."""
        client, mock_instance = mock_client

        # Mock successful responses
        mock_response = Mock()
        mock_response.content = [Mock(text='{"result": "success"}')]
        mock_response.usage.total_tokens = 100
        mock_instance.messages.create.return_value = mock_response

        # Test batch analysis
        documents = [
            {"content": "doc1", "file_name": "file1.txt"},
            {"content": "doc2", "file_name": "file2.txt"},
        ]

        results = client.analyze_batch(documents, batch_size=2)

        assert len(results) == 2
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert mock_instance.messages.create.call_count == 2

    def test_analyze_batch_with_error(self, mock_client):
        """Test batch analysis with error handling."""
        client, mock_instance = mock_client

        # Mock one success and one failure
        mock_response = Mock()
        mock_response.content = [Mock(text='{"result": "success"}')]
        mock_response.usage.total_tokens = 100

        mock_instance.messages.create.side_effect = [
            mock_response,
            Exception("API Error"),
        ]

        documents = [
            {"content": "doc1", "file_name": "file1.txt"},
            {"content": "doc2", "file_name": "file2.txt"},
        ]

        results = client.analyze_batch(documents)

        assert len(results) == 2
        assert results[0].metadata.get("error") is None
        assert results[1].metadata.get("error") is True
        assert results[1].confidence_score == 0.0

    def test_validate_connection(self, mock_client):
        """Test connection validation."""
        client, mock_instance = mock_client

        # Mock successful response
        mock_response = Mock()
        mock_instance.messages.create.return_value = mock_response

        assert client.validate_connection() is True

        # Mock error response
        mock_instance.messages.create.side_effect = Exception("Connection error")
        assert client.validate_connection() is False

    def test_get_model_info(self, mock_client):
        """Test getting model information."""
        client, _ = mock_client

        info = client.get_model_info()

        assert info["model"] == "claude-3-7-sonnet-20250219"
        assert info["max_tokens"] == 4000
        assert info["temperature"] == 0.7
        assert info["retry_attempts"] == 3


class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_select_prompt_by_file_type(self):
        """Test prompt selection based on file type."""
        pdf_prompt = PromptTemplates.select_prompt("pdf")
        assert "PDF document" in pdf_prompt

        image_prompt = PromptTemplates.select_prompt("jpg")
        assert "image of a document" in image_prompt

        email_prompt = PromptTemplates.select_prompt("eml")
        assert "email" in email_prompt

    def test_select_prompt_by_content(self):
        """Test prompt selection based on content."""
        financial_content = "Invoice: $1000 payment due"
        prompt = PromptTemplates.select_prompt("txt", financial_content)
        assert "financial document" in prompt

        legal_content = "This agreement is hereby entered"
        prompt = PromptTemplates.select_prompt("txt", legal_content)
        assert "legal document" in prompt

        medical_content = "Patient diagnosis and prescription"
        prompt = PromptTemplates.select_prompt("txt", medical_content)
        assert "medical document" in prompt

    def test_format_prompt(self):
        """Test prompt formatting."""
        template = "Analyze {content} from {file_name}"
        formatted = PromptTemplates.format_prompt(
            template, content="test content", file_name="test.txt"
        )

        assert formatted == "Analyze test content from test.txt"

    def test_default_prompt_structure(self):
        """Test default prompt has required structure."""
        prompt = PromptTemplates.get_default_analysis_prompt()

        # Check for required fields in JSON structure
        assert '"document_type"' in prompt
        assert '"category"' in prompt
        assert '"confidence_score"' in prompt
        assert '"suggested_folder"' in prompt
