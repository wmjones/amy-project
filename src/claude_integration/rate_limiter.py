"""
Rate limiting utilities for Claude API.
"""

import time
import threading
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for permission

        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()

        while True:
            with self.lock:
                now = time.time()

                # Remove old requests outside the window
                while self.requests and self.requests[0] <= now - self.window_seconds:
                    self.requests.popleft()

                # Check if we can make a request
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True

                # Calculate wait time
                if self.requests:
                    oldest_request = self.requests[0]
                    wait_time = (oldest_request + self.window_seconds) - now
                else:
                    wait_time = 0

            # Check timeout
            if timeout is not None and time.time() - start_time + wait_time > timeout:
                return False

            # Wait before retry
            time.sleep(min(wait_time, 0.1))

    def reset(self):
        """Reset the rate limiter."""
        with self.lock:
            self.requests.clear()

    def get_status(self) -> dict:
        """Get current rate limiter status.

        Returns:
            Dictionary with status information
        """
        with self.lock:
            now = time.time()

            # Clean old requests
            while self.requests and self.requests[0] <= now - self.window_seconds:
                self.requests.popleft()

            requests_count = len(self.requests)

            # Calculate time until next available slot
            if requests_count >= self.max_requests and self.requests:
                oldest_request = self.requests[0]
                wait_time = max(0, (oldest_request + self.window_seconds) - now)
            else:
                wait_time = 0

            return {
                "current_requests": requests_count,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "available_requests": max(0, self.max_requests - requests_count),
                "wait_time_seconds": wait_time,
            }


class TokenBucket:
    """Token bucket rate limiter for smoother rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_and_consume(
        self, tokens: int = 1, timeout: Optional[float] = None
    ) -> bool:
        """Wait for tokens to become available and consume them.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait

        Returns:
            True if tokens were consumed, False if timeout
        """
        start_time = time.time()

        while True:
            if self.consume(tokens):
                return True

            with self.lock:
                wait_time = (tokens - self.tokens) / self.refill_rate

            if timeout is not None and time.time() - start_time + wait_time > timeout:
                return False

            time.sleep(min(wait_time, 0.1))

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill

        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def get_status(self) -> dict:
        """Get current token bucket status.

        Returns:
            Dictionary with status information
        """
        with self.lock:
            self._refill()

            return {
                "current_tokens": self.tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "tokens_per_minute": self.refill_rate * 60,
            }
