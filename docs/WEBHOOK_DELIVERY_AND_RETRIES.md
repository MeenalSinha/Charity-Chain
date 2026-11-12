# CharityChain — Webhook Delivery, Retries & Dead-Letter Handling

## 1. Overview

CharityChain guarantees at-least-once delivery for outbound webhook events using an exponential backoff retry scheduler with full random jitter, explicit attempt telemetry, and circuit-breaker style auto-suspension.

---

## 2. Retry Policy & Exponential Backoff with Jitter

When a webhook destination endpoint fails to return an HTTP `2xx` response (or times out after 10 seconds), the delivery engine schedules a retry.

### Backoff Calculation
The delay before attempt $n$ is calculated as:

$$\text{delay} = \min(\text{max\_seconds}, \text{base\_seconds} \times 2^{n-1}) + \text{jitter}$$

Where:
- $\text{base\_seconds} = 10.0$
- $\text{max\_seconds} = 86400.0$ (24 hours)
- $\text{jitter} \sim \text{Uniform}(0, \min(10.0, 0.2 \times \text{exponential}))$

### Retry Timeline Example
| Attempt | Base Delay | Random Jitter | Next Retry Interval |
|---|---|---|---|
| Attempt 1 | 10 seconds | 0–2.0s | ~10–12 seconds |
| Attempt 2 | 20 seconds | 0–4.0s | ~20–24 seconds |
| Attempt 3 | 40 seconds | 0–8.0s | ~40–48 seconds |
| Attempt 4 | 80 seconds | 0–10.0s | ~80–90 seconds |
| Attempt 5 | 160 seconds | 0–10.0s | ~160–170 seconds |

---

## 3. Dead-Letter Queue (DLQ) State

If a webhook delivery fails all 5 retry attempts, its status transitions to `DEAD_LETTER`:

```
PENDING ──► PROCESSING ──► FAILED (Attempt 1..4) ──► DEAD_LETTER (Attempt 5)
```

Dead-lettered deliveries:
- Halt automated retries.
- Maintain full diagnostic records in `webhook_deliveries` and `webhook_delivery_attempts`.
- Can be manually re-triggered via the administrative API (`POST /api/v1/webhooks/deliveries/{id}/retry`).

---

## 4. Automatic Subscription Suspension

If an endpoint fails repeatedly across distinct events, the system automatically suspends the subscription to conserve platform resources and alert the subscriber:

- **Threshold**: 10 consecutive delivery failures (`WEBHOOK_AUTO_SUSPEND_THRESHOLD = 10`).
- **Effect**: `WebhookSubscription.is_active` set to `False`, `suspended_at` recorded.
- **Resolution**: Subscriber inspects logs, resolves endpoint issue, and calls `POST /api/v1/webhooks/subscriptions/{id}/reactivate`.

