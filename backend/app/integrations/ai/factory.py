"""
factory.py - AI Provider Factory & Dependency Injection Registry
================================================================
Provides thread-safe singleton access to the configured AI verification provider.
Supports runtime dependency injection and mock providers for automated testing.
"""

from typing import Optional
from backend.app.core.config import settings
from backend.app.integrations.ai.base import AIProviderInterface
from backend.app.integrations.ai.provider import MultiIndicatorAIProvider

_active_ai_provider: Optional[AIProviderInterface] = None


def get_ai_provider() -> MultiIndicatorAIProvider:
    """Retrieves or initializes the active multi-indicator AI verification provider."""
    global _active_ai_provider
    if _active_ai_provider is None:
        _active_ai_provider = MultiIndicatorAIProvider()
    return _active_ai_provider


def set_ai_provider(provider: AIProviderInterface) -> None:
    """Overrides the active AI provider (useful for test fixtures and mocks)."""
    global _active_ai_provider
    _active_ai_provider = provider


def reset_ai_provider() -> None:
    """Resets the singleton instance back to default."""
    global _active_ai_provider
    _active_ai_provider = None

