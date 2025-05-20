"""
Configuration for Claude integration.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ClaudeConfig:
    """Configuration for Claude API client."""

    # API Configuration
    api_key: str
    model: str = "claude-3-7-sonnet-20250219"
    max_tokens: int = 4000
    temperature: float = 0.7

    # Retry Configuration
    retry_attempts: int = 3
    retry_delay: float = 2.0

    # Rate Limiting
    requests_per_minute: int = 60
    batch_size: int = 5

    # Analysis Configuration
    confidence_threshold: float = 0.7
    max_content_length: int = 10000

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            model=os.getenv("MODEL", cls.model),
            max_tokens=int(os.getenv("MAX_TOKENS", cls.max_tokens)),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", cls.retry_attempts)),
            retry_delay=float(os.getenv("RETRY_DELAY", cls.retry_delay)),
            batch_size=int(os.getenv("BATCH_SIZE", cls.batch_size)),
            confidence_threshold=float(
                os.getenv("MIN_CONFIDENCE_SCORE", cls.confidence_threshold)
            ),
        )

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api_key:
            return False

        if self.max_tokens < 1 or self.max_tokens > 200000:
            return False

        if self.temperature < 0 or self.temperature > 1:
            return False

        if self.retry_attempts < 0:
            return False

        return True


class ModelProfiles:
    """Predefined model profiles for different use cases."""

    # Fast, low-cost analysis
    FAST = ClaudeConfig(
        api_key="",  # Will be set from env
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        temperature=0.3,
        retry_attempts=2,
    )

    # Balanced performance
    BALANCED = ClaudeConfig(
        api_key="",
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0.5,
        retry_attempts=3,
    )

    # High accuracy analysis
    ACCURATE = ClaudeConfig(
        api_key="",
        model="claude-3-7-sonnet-20250219",
        max_tokens=8000,
        temperature=0.7,
        retry_attempts=5,
    )

    @classmethod
    def get_profile(cls, profile_name: str) -> ClaudeConfig:
        """Get a predefined profile by name."""
        profiles = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "accurate": cls.ACCURATE,
        }

        profile = profiles.get(profile_name.lower())
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        # Set API key from environment
        profile.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        return profile
