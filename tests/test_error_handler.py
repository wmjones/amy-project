"""
Unit tests for the error handler module.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

import anthropic
import requests

from src.utils.error_handler import ErrorHandler, ErrorType, ErrorSeverity, ErrorRecord


class TestErrorHandler:
    """Test the ErrorHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()

    def test_error_categorization(self):
        """Test error type categorization."""
        # Test transient errors
        connection_error = ConnectionError("Connection failed")
        assert (
            self.error_handler._categorize_error(connection_error)
            == ErrorType.TRANSIENT
        )

        timeout_error = TimeoutError("Timeout occurred")
        assert (
            self.error_handler._categorize_error(timeout_error) == ErrorType.TRANSIENT
        )

        # Test file access errors
        file_error = FileNotFoundError("File not found")
        assert self.error_handler._categorize_error(file_error) == ErrorType.FILE_ACCESS

        permission_error = PermissionError("Permission denied")
        assert (
            self.error_handler._categorize_error(permission_error)
            == ErrorType.FILE_ACCESS
        )

        # Test API errors
        mock_http_request = Mock()
        api_error = anthropic.APIError(
            "API error", request=mock_http_request, body=None
        )
        assert self.error_handler._categorize_error(api_error) == ErrorType.API

        mock_http_response = Mock(headers={}, status_code=429)
        rate_limit_error = anthropic.RateLimitError(
            "Rate limit exceeded", response=mock_http_response, body=None
        )
        assert self.error_handler._categorize_error(rate_limit_error) == ErrorType.API

        # Test processing errors
        value_error = ValueError("Invalid value")
        assert self.error_handler._categorize_error(value_error) == ErrorType.PROCESSING

        # Test configuration errors
        import_error = ImportError("Module not found")
        assert (
            self.error_handler._categorize_error(import_error)
            == ErrorType.CONFIGURATION
        )

        # Test unknown errors
        custom_error = Exception("Custom error")
        assert self.error_handler._categorize_error(custom_error) == ErrorType.UNKNOWN

    def test_severity_determination(self):
        """Test error severity determination."""
        # Test transient error severity
        connection_error = ConnectionError("Connection failed")
        assert (
            self.error_handler._determine_severity(
                connection_error, ErrorType.TRANSIENT
            )
            == ErrorSeverity.LOW
        )

        # Test API error severity
        mock_http_response = Mock(headers={}, status_code=429)
        rate_limit_error = anthropic.RateLimitError(
            "Rate limit exceeded", response=mock_http_response, body=None
        )
        assert (
            self.error_handler._determine_severity(rate_limit_error, ErrorType.API)
            == ErrorSeverity.MEDIUM
        )

        mock_http_request = Mock()
        api_error = anthropic.APIError(
            "API error", request=mock_http_request, body=None
        )
        assert (
            self.error_handler._determine_severity(api_error, ErrorType.API)
            == ErrorSeverity.HIGH
        )

        # Test file access error severity
        permission_error = PermissionError("Permission denied")
        assert (
            self.error_handler._determine_severity(
                permission_error, ErrorType.FILE_ACCESS
            )
            == ErrorSeverity.HIGH
        )

        file_error = FileNotFoundError("File not found")
        assert (
            self.error_handler._determine_severity(file_error, ErrorType.FILE_ACCESS)
            == ErrorSeverity.MEDIUM
        )

        # Test configuration error severity
        import_error = ImportError("Module not found")
        assert (
            self.error_handler._determine_severity(
                import_error, ErrorType.CONFIGURATION
            )
            == ErrorSeverity.CRITICAL
        )

    def test_retry_operation_success(self):
        """Test successful retry operation."""
        # Mock function that fails twice then succeeds
        call_count = 0

        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        self.error_handler.max_retries = 3
        self.error_handler.initial_backoff = 0.01  # Fast for testing

        result, error = self.error_handler._retry_operation(mock_func, "test_context")

        assert result == "success"
        assert error is None
        assert call_count == 3

    def test_retry_operation_failure(self):
        """Test retry operation that exhausts retries."""

        # Mock function that always fails
        def mock_func():
            raise ConnectionError("Connection failed")

        self.error_handler.max_retries = 2
        self.error_handler.initial_backoff = 0.01  # Fast for testing

        result, error = self.error_handler._retry_operation(mock_func, "test_context")

        assert result is None
        assert isinstance(error, ConnectionError)

    def test_handle_error_with_retry(self):
        """Test error handling with retry functionality."""
        # Mock retry function
        retry_func = Mock(return_value="success")

        # Test transient error with retry
        error = ConnectionError("Connection failed")
        self.error_handler.max_retries = 1
        self.error_handler.initial_backoff = 0.01

        result, err = self.error_handler.handle_error(error, "test_context", retry_func)

        assert result == "success"
        assert err is None
        retry_func.assert_called_once()

    def test_handle_file_access_error(self):
        """Test file access error handling."""
        # Test FileNotFoundError
        error = FileNotFoundError("File not found")
        result, err = self.error_handler.handle_error(error, "test_context")

        assert result is None
        assert err == error

        # Test PermissionError
        error = PermissionError("Permission denied")
        result, err = self.error_handler.handle_error(error, "test_context")

        assert result is None
        assert err == error

    def test_handle_api_error(self):
        """Test API error handling."""
        # Test regular API error
        mock_http_request = Mock()
        error = anthropic.APIError("API error", request=mock_http_request, body=None)
        result, err = self.error_handler.handle_error(error, "test_context")

        assert result is None
        assert err == error

        # Test rate limit error with retry function
        mock_http_response = Mock(headers={}, status_code=429)
        error = anthropic.RateLimitError(
            "Rate limit exceeded", response=mock_http_response, body=None
        )
        error.retry_after = 0.01  # Fast for testing
        retry_func = Mock(return_value="success")

        result, err = self.error_handler.handle_error(error, "test_context", retry_func)

        assert result == "success"
        assert err is None
        retry_func.assert_called_once()

    def test_error_history_tracking(self):
        """Test error history tracking."""
        # Generate some errors
        errors = [
            ConnectionError("Error 1"),
            FileNotFoundError("Error 2"),
            ValueError("Error 3"),
        ]

        for error in errors:
            self.error_handler.handle_error(error, f"context_{error}")

        # Check error history
        assert len(self.error_handler.error_history) == 3
        assert all(
            isinstance(record, ErrorRecord)
            for record in self.error_handler.error_history
        )

        # Check error counts
        assert self.error_handler.error_counts[ErrorType.TRANSIENT] == 1
        assert self.error_handler.error_counts[ErrorType.FILE_ACCESS] == 1
        assert self.error_handler.error_counts[ErrorType.PROCESSING] == 1

    def test_error_statistics(self):
        """Test error statistics generation."""
        # Generate some errors
        self.error_handler.handle_error(ConnectionError("Error 1"), "context_1")
        self.error_handler.handle_error(FileNotFoundError("Error 2"), "context_2")

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == 2
        assert stats["error_counts_by_type"]["transient"] == 1
        assert stats["error_counts_by_type"]["file_access"] == 1
        assert len(stats["recent_errors"]) == 2

    def test_save_error_report(self, tmp_path):
        """Test saving error report to file."""
        # Generate some errors
        self.error_handler.handle_error(ConnectionError("Error 1"), "context_1")
        self.error_handler.handle_error(FileNotFoundError("Error 2"), "context_2")

        # Save report
        report_path = tmp_path / "error_report.json"
        self.error_handler.save_error_report(report_path)

        # Verify report exists and contains data
        assert report_path.exists()

        import json

        with open(report_path) as f:
            report = json.load(f)

        assert "generated_at" in report
        assert "statistics" in report
        assert "error_history" in report
        assert report["statistics"]["total_errors"] == 2

    def test_error_record(self):
        """Test ErrorRecord class."""
        error = ValueError("Test error")
        record = ErrorRecord(
            error=error,
            context="test_context",
            error_type=ErrorType.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
        )

        # Test to_dict
        error_dict = record.to_dict()
        assert error_dict["error"] == "Test error"
        assert error_dict["context"] == "test_context"
        assert error_dict["error_type"] == "processing"
        assert error_dict["severity"] == "medium"
        assert "timestamp" in error_dict
        assert "traceback" in error_dict


if __name__ == "__main__":
    pytest.main([__file__])
