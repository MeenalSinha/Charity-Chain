"""
test_rate_limiter.py - Unit Tests for Sliding-Window Rate Limiter
=================================================================
Validates rate limit calculations, window pruning, response headers,
and HTTP 429 rejection mechanics.
"""

import time
import pytest
from unittest.mock import MagicMock
from fastapi import Request, Response, status

from backend.app.core.rate_limiter import (
    SlidingWindowRateLimiter,
    RateLimitExceededException,
    RateLimiterDependency,
    get_client_identifier,
    limiter,
)
from backend.app.core.config import settings


@pytest.fixture(autouse=True)
def clean_rate_limiter():
    """Ensure in-memory rate limiter windows are clean before and after each test."""
    limiter.reset_in_memory()
    yield
    limiter.reset_in_memory()


def test_sliding_window_allows_within_limit():
    """Verifies that requests below the threshold are permitted."""
    test_limiter = SlidingWindowRateLimiter(redis_client=None)  # Uses memory fallback
    key = "test_user_1"
    limit = 3
    window = 10

    # 1st request
    allowed, count, remaining, reset = test_limiter.check_rate_limit(key, limit, window)
    assert allowed is True
    assert count == 1
    assert remaining == 2

    # 2nd request
    allowed, count, remaining, reset = test_limiter.check_rate_limit(key, limit, window)
    assert allowed is True
    assert count == 2
    assert remaining == 1

    # 3rd request (reaching limit)
    allowed, count, remaining, reset = test_limiter.check_rate_limit(key, limit, window)
    assert allowed is True
    assert count == 3
    assert remaining == 0


def test_sliding_window_rejects_exceeding_limit():
    """Verifies that requests exceeding the limit are blocked with 0 remaining."""
    test_limiter = SlidingWindowRateLimiter(redis_client=None)
    key = "test_user_blocked"
    limit = 2
    window = 10

    # Fill capacity
    test_limiter.check_rate_limit(key, limit, window)
    test_limiter.check_rate_limit(key, limit, window)

    # 3rd request must be blocked
    allowed, count, remaining, reset = test_limiter.check_rate_limit(key, limit, window)
    assert allowed is False
    assert count == 2
    assert remaining == 0
    assert reset > 0


def test_client_identifier_prioritizes_authenticated_user():
    """Verifies user ID takes precedence over IP addresses."""
    request = MagicMock(spec=Request)
    user_mock = MagicMock()
    user_mock.id = "user-uuid-123"
    request.state.user = user_mock
    request.headers = {"X-Forwarded-For": "203.0.113.195"}

    identifier = get_client_identifier(request)
    assert identifier == "user:user-uuid-123"


def test_client_identifier_uses_forwarded_ip_when_unauthenticated():
    """Verifies forwarded IP is used when no user is authenticated."""
    request = MagicMock(spec=Request)
    request.state.user = None
    request.headers = {"X-Forwarded-For": "198.51.100.42, 10.0.0.1"}

    identifier = get_client_identifier(request)
    assert identifier == "ip:198.51.100.42"


def test_rate_limiter_dependency_raises_429():
    """Verifies that RateLimiterDependency raises RateLimitExceededException when throttled."""
    dep = RateLimiterDependency(limit=1, window_seconds=60, tier_name="test_dep")
    request = MagicMock(spec=Request)
    request.state.user = None
    request.headers = {}
    request.client.host = "192.0.2.1"
    response = Response()

    # First call succeeds
    dep(request, response)
    assert response.headers["X-RateLimit-Limit"] == "1"
    assert response.headers["X-RateLimit-Remaining"] == "0"

    # Second call must raise 429
    with pytest.raises(RateLimitExceededException) as exc_info:
        dep(request, response)

    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Retry-After" in exc_info.value.headers
