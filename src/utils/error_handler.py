"""
Error handling and recovery mechanisms for the file organization system.
Provides robust error handling, categorization, retry logic, and recovery strategies.
"""

import logging
import time
import traceback
from typing import Optional, Callable, Any, Tuple, Dict, List
from datetime import datetime
from pathlib import Path
from enum import Enum

import anthropic
import requests

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Categorization of different error types."""

    TRANSIENT = "transient"
    FILE_ACCESS = "file_access"
    API = "api"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorRecord:
    """Record of an error occurrence."""

    def __init__(
        self,
        error: Exception,
        context: str,
        error_type: ErrorType,
        severity: ErrorSeverity,
    ):
        self.error = error
        self.context = context
        self.error_type = error_type
        self.severity = severity
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "error": str(self.error),
            "context": self.context,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback,
        }


class ErrorHandler:
    """Handle errors with retry logic and recovery strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ):
        """
        Initialize error handler.

        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.logger = logging.getLogger("error_handler")
        self.error_history: List[ErrorRecord] = []
        self.error_counts: Dict[ErrorType, int] = {
            error_type: 0 for error_type in ErrorType
        }

    def handle_error(
        self, error: Exception, context: str, retry_func: Optional[Callable] = None
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error: The exception to handle
            context: Context describing where the error occurred
            retry_func: Optional function to retry

        Returns:
            Tuple of (result, error) where result is the successful result if any
        """
        error_type = self._categorize_error(error)
        severity = self._determine_severity(error, error_type)

        # Record the error
        error_record = ErrorRecord(error, context, error_type, severity)
        self.error_history.append(error_record)
        self.error_counts[error_type] += 1

        self.logger.error(
            f"Error in {context}: {str(error)}",
            extra={"error_type": error_type.value, "severity": severity.value},
        )

        # Handle based on error type
        if error_type == ErrorType.TRANSIENT and retry_func:
            return self._retry_operation(retry_func, context)
        elif error_type == ErrorType.FILE_ACCESS:
            return self._handle_file_access_error(error, context)
        elif error_type == ErrorType.API:
            return self._handle_api_error(error, context, retry_func)
        elif error_type == ErrorType.PROCESSING:
            return self._handle_processing_error(error, context)
        elif error_type == ErrorType.CONFIGURATION:
            return self._handle_configuration_error(error, context)
        else:
            return self._handle_unknown_error(error, context)

    def _categorize_error(self, error: Exception) -> ErrorType:
        """Categorize the error type."""
        if isinstance(
            error,
            (
                ConnectionError,
                TimeoutError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        ):
            return ErrorType.TRANSIENT
        elif isinstance(error, (FileNotFoundError, PermissionError, OSError, IOError)):
            return ErrorType.FILE_ACCESS
        elif isinstance(error, (anthropic.APIError, anthropic.RateLimitError)):
            return ErrorType.API
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorType.PROCESSING
        elif isinstance(error, (AttributeError, ImportError)):
            return ErrorType.CONFIGURATION
        else:
            return ErrorType.UNKNOWN

    def _determine_severity(
        self, error: Exception, error_type: ErrorType
    ) -> ErrorSeverity:
        """Determine error severity based on error type and specifics."""
        if error_type == ErrorType.TRANSIENT:
            return ErrorSeverity.LOW
        elif error_type == ErrorType.API:
            if isinstance(error, anthropic.RateLimitError):
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.HIGH
        elif error_type == ErrorType.FILE_ACCESS:
            if isinstance(error, PermissionError):
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.MEDIUM
        elif error_type == ErrorType.CONFIGURATION:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.HIGH

    def _retry_operation(
        self, retry_func: Callable, context: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Retry an operation with exponential backoff."""
        for attempt in range(self.max_retries):
            backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
            self.logger.info(
                f"Retrying {context} in {backoff:.1f} seconds "
                f"(attempt {attempt+1}/{self.max_retries})"
            )
            time.sleep(backoff)

            try:
                result = retry_func()
                self.logger.info(f"Retry successful for {context}")
                return result, None
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"All retry attempts failed for {context}: {str(e)}"
                    )
                    return None, e

        return None, Exception(f"Retry attempts exhausted for {context}")

    def _handle_file_access_error(
        self, error: Exception, context: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Handle file access errors."""
        if isinstance(error, FileNotFoundError):
            self.logger.warning(f"File not found in {context}: {str(error)}")
            return None, error
        elif isinstance(error, PermissionError):
            self.logger.error(f"Permission denied in {context}: {str(error)}")
            # Could attempt to change permissions or skip
            return None, error
        else:
            self.logger.error(f"File access error in {context}: {str(error)}")
            return None, error

    def _handle_api_error(
        self, error: Exception, context: str, retry_func: Optional[Callable] = None
    ) -> Tuple[Any, Optional[Exception]]:
        """Handle API-specific errors."""
        if isinstance(error, anthropic.RateLimitError):
            # Wait for rate limit to reset
            wait_time = getattr(error, "retry_after", 60)
            self.logger.warning(
                f"Rate limit reached in {context}. Waiting {wait_time} seconds."
            )
            time.sleep(wait_time)
            if retry_func:
                return retry_func(), None
            return None, error
        else:
            self.logger.error(f"API error in {context}: {str(error)}")
            return None, error

    def _handle_processing_error(
        self, error: Exception, context: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Handle processing errors."""
        self.logger.error(f"Processing error in {context}: {str(error)}")
        # Log the full traceback for debugging
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, error

    def _handle_configuration_error(
        self, error: Exception, context: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Handle configuration errors."""
        self.logger.critical(f"Configuration error in {context}: {str(error)}")
        # These are typically unrecoverable
        return None, error

    def _handle_unknown_error(
        self, error: Exception, context: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Handle unknown errors."""
        self.logger.error(f"Unknown error in {context}: {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered."""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_type": {
                error_type.value: count
                for error_type, count in self.error_counts.items()
            },
            "recent_errors": [error.to_dict() for error in self.error_history[-10:]],
        }

    def save_error_report(self, filepath: Path):
        """Save a detailed error report to file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "error_history": [error.to_dict() for error in self.error_history],
        }

        import json

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Error report saved to {filepath}")
