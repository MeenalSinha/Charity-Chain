"""
rate_limiter.py - Distributed Sliding-Window Rate Limiter & Abuse Protection
=============================================================================
Enforces granular rate limits across API endpoints using Redis sorted sets (ZSET),
with an automated thread-safe in-memory fallback for offline/test environments.
Emits standard RFC 6585 headers:
  - X-RateLimit-Limit
  - X-RateLimit-Remaining
  - X-RateLimit-Reset
  - Retry-After
"""

import time
import uuid
import logging
import threading
from typing import Optional, Tuple, Dict, Deque
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status, Response

from backend.app.core.config import settings

logger = logging.getLogger("charitychain.rate_limiter")

# In-memory sliding-window storage fallback
_in_memory_windows: Dict[str, Deque[float]] = defaultdict(deque)
_in_memory_lock = threading.Lock()


class RateLimitExceededException(HTTPException):
    """Raised when an incoming request exceeds the configured rate limit."""

    def __init__(self, limit: int, retry_after: int, message: str = "Too many requests. Please try again later."):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": message,
                "limit": limit,
                "retry_after_seconds": retry_after,
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
                "Retry-After": str(retry_after),
            },
        )


class SlidingWindowRateLimiter:
    """Distributed Redis-backed sliding window rate limiter with in-memory fallback."""

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            return self._redis
        except Exception as e:
            logger.debug(f"Redis not available for rate limiting, using memory fallback: {e}")
            return None

    def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, int, int]:
        """
        Check and record a request against the sliding window.
        Returns:
            (allowed: bool, count: int, remaining: int, reset_after: int)
        """
        now = time.time()
        window_start = now - window_seconds
        r = self._get_redis()

        if r is not None:
            try:
                pipe = r.pipeline()
                redis_key = f"rl:{key}"
                member_id = f"{now:.6f}:{uuid.uuid4().hex[:8]}"

                # 1. Remove expired timestamps outside the sliding window
                pipe.zremrangebyscore(redis_key, 0, window_start)
                # 2. Count current elements in window
                pipe.zcard(redis_key)
                # 3. Oldest element in window to calculate accurate reset
                pipe.zrange(redis_key, 0, 0, withscores=True)
                
                results = pipe.execute()
                current_count = results[1]
                oldest_entry = results[2]

                if current_count >= limit:
                    if oldest_entry:
                        oldest_ts = oldest_entry[0][1]
                        reset_after = max(1, int(oldest_ts + window_seconds - now))
                    else:
                        reset_after = window_seconds
                    return False, current_count, 0, reset_after

                # Below limit: record request and refresh TTL
                pipe = r.pipeline()
                pipe.zadd(redis_key, {member_id: now})
                pipe.expire(redis_key, window_seconds + 5)
                pipe.execute()

                remaining = max(0, limit - (current_count + 1))
                return True, current_count + 1, remaining, window_seconds

            except Exception as exc:
                logger.warning(f"Redis rate limiting failed ({exc}); falling back to in-memory window")

        # In-memory thread-safe fallback
        with _in_memory_lock:
            q = _in_memory_windows[key]
            while q and q[0] <= window_start:
                q.popleft()

            current_count = len(q)
            if current_count >= limit:
                oldest_ts = q[0]
                reset_after = max(1, int(oldest_ts + window_seconds - now))
                return False, current_count, 0, reset_after

            q.append(now)
            remaining = max(0, limit - len(q))
            return True, len(q), remaining, window_seconds

    def reset_in_memory(self):
        """Clears in-memory rate limiting windows (used for test teardown)."""
        with _in_memory_lock:
            _in_memory_windows.clear()


# Global singleton rate limiter instance
limiter = SlidingWindowRateLimiter()


def get_client_identifier(request: Request) -> str:
    """
    Extracts a high-entropy identifier for rate limiting:
    Prioritizes authenticated user ID, then forwarded client IP, then direct host.
    """
    # 1. Authenticated User (if present on request.state)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id") and user.id:
        return f"user:{user.id}"

    # 2. X-Forwarded-For header (first hop)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return f"ip:{client_ip}"

    # 3. Direct client host
    if request.client and request.client.host:
        return f"ip:{request.client.host}"

    return "ip:unknown"


class RateLimiterDependency:
    """FastAPI Dependency for declarative endpoint rate limiting."""

    def __init__(self, limit: int, window_seconds: int = 60, tier_name: str = "default"):
        self.limit = limit
        self.window_seconds = window_seconds
        self.tier_name = tier_name

    def __call__(self, request: Request, response: Response):
        if not settings.RATE_LIMITING_ENABLED:
            return

        client_id = get_client_identifier(request)
        key = f"{self.tier_name}:{client_id}"

        allowed, count, remaining, reset_after = limiter.check_rate_limit(
            key=key,
            limit=self.limit,
            window_seconds=self.window_seconds,
        )

        # Attach standard rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_after)

        if not allowed:
            response.headers["Retry-After"] = str(reset_after)
            logger.warning(
                f"Rate limit exceeded: tier={self.tier_name} client={client_id} count={count}/{self.limit}"
            )
            raise RateLimitExceededException(limit=self.limit, retry_after=reset_after)


# Convenience factories for standard endpoint tiers
def auth_rate_limit(limit: Optional[int] = None, window_seconds: int = 60):
    effective_limit = limit or settings.RATE_LIMIT_AUTH_PER_MINUTE
    return RateLimiterDependency(limit=effective_limit, window_seconds=window_seconds, tier_name="auth")


def financial_rate_limit(limit: Optional[int] = None, window_seconds: int = 60):
    effective_limit = limit or settings.RATE_LIMIT_FINANCIAL_PER_MINUTE
    return RateLimiterDependency(limit=effective_limit, window_seconds=window_seconds, tier_name="financial")


def evidence_rate_limit(limit: Optional[int] = None, window_seconds: int = 60):
    effective_limit = limit or settings.RATE_LIMIT_EVIDENCE_PER_MINUTE
    return RateLimiterDependency(limit=effective_limit, window_seconds=window_seconds, tier_name="evidence")


def default_rate_limit(limit: Optional[int] = None, window_seconds: int = 60):
    effective_limit = limit or settings.RATE_LIMIT_DEFAULT_PER_MINUTE
    return RateLimiterDependency(limit=effective_limit, window_seconds=window_seconds, tier_name="default")
