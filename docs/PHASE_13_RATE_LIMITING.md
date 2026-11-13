# CharityChain — Phase 13 Rate Limiting & Abuse Protection
**Document Version:** 1.0.0  
**Implementation File:** `backend/app/core/rate_limiter.py`  
**Architecture:** Distributed Redis Sorted Set (`ZSET`) Sliding Window with In-Memory Fallback  

---

## 1. Overview & Objectives

API rate limiting protects CharityChain against credential stuffing, brute-force password guessing, payment card testing, resource exhaustion in AI evidence processing, and scraping of sensitive NGO data.

The sliding window algorithm guarantees that traffic spikes spanning window boundaries (e.g., at minute roll-overs) cannot exceed the rate limit ceiling.

---

## 2. Sliding Window Algorithm Implementation

### 2.1 Redis Sorted Set Mechanics
Each rate-limited endpoint creates a Redis key:
`rl:{tier_name}:{identifier}`

Where `identifier` is:
1. `user:{user_id}` (if user is authenticated)
2. `ip:{client_ip}` (from `X-Forwarded-For` or socket address)

```
Time: ------------------------[ Window Start -----------> Now ]
Score (Timestamp):            [ 1700000010, 1700000025, 1700000055 ]
Elements outside window are purged via ZREMRANGEBYSCORE.
Remaining elements are counted via ZCARD.
```

### 2.2 Atomic Operations Pipeline
```python
pipe = redis.pipeline()
pipe.zremrangebyscore(key, 0, now - window_seconds)
pipe.zcard(key)
pipe.zrange(key, 0, 0, withscores=True) # Oldest entry for accurate reset calculation
results = pipe.execute()
```

If count < limit:
```python
pipe = redis.pipeline()
pipe.zadd(key, {unique_member_id: now})
pipe.expire(key, window_seconds + 5)
pipe.execute()
```

---

## 3. Rate Limit Tiers & Configurations

| Tier | Default Limit | Window | Target Endpoints | Protection Goal |
|---|:---:|:---:|---|---|
| **Auth** | 5 requests | 60 sec | `/api/v1/auth/login`, `/register`, `/forgot-password` | Prevents credential stuffing, account enumeration |
| **Financial** | 20 requests | 60 sec | `/api/v1/donations/*`, `/payouts/*` | Prevents payment intent abuse and card testing |
| **Evidence & AI** | 10 requests | 60 sec | `/api/v1/evidence/*` | Prevents GPU/CPU starvation via forensic image flooding |
| **Default** | 100 requests | 60 sec | General API endpoints | General service availability |

---

## 4. HTTP Headers & Error Response Contract

### 4.1 Response Headers
Every evaluated response receives standard rate limit telemetry headers:
```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 19
X-RateLimit-Reset: 60
```

When the limit is breached:
```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 42
Retry-After: 42

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "limit": 20,
  "retry_after_seconds": 42
}
```

---

## 5. Offline & High-Availability Resilience

If the Redis cluster is unreachable or offline (e.g. during standalone testing), `SlidingWindowRateLimiter` seamlessly falls back to an in-memory thread-safe `deque` window protected by a reentrant mutex lock, logging a warning while maintaining continuous application availability.
