# CharityChain — In-App Notification Center & Preferences API

## 1. Overview

CharityChain provides a secure, user-isolated In-App Notification Center where donors, NGOs, and reviewers track milestones, approvals, payouts, and certificates.

---

## 2. API Endpoints

All endpoints are mounted under `/api/v1/notifications` and require valid JWT authentication.

### `GET /api/v1/notifications`
Lists notifications for the authenticated user with optional filtering and pagination.

- **Query Parameters**:
  - `unread_only` (bool, default: `false`): Filter by unread status.
  - `include_archived` (bool, default: `false`): Include archived notifications.
  - `category` (string, optional): Filter by `NotificationCategory` (`DONATIONS`, `PROJECTS`, `MILESTONES`, `EVIDENCE`, `VERIFICATION`, `DISBURSEMENTS`, `CERTIFICATES`, `SECURITY`).
  - `page` (int, default: 1): Page number.
  - `page_size` (int, default: 20): Items per page (max: 100).

- **Response (200 OK)**:
```json
{
  "total": 42,
  "unread_count": 3,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "id": "notif-uuid-123",
      "category": "DONATIONS",
      "event_type": "payment.captured",
      "title": "Donation Payment Confirmed",
      "message": "Contribution of $100.00 confirmed for Clean Water Wells.",
      "action_url": "https://charitychain.org/donor/donations",
      "is_read": false,
      "read_at": null,
      "is_archived": false,
      "created_at": "2026-11-12T22:30:00Z"
    }
  ]
}
```

---

### `GET /api/v1/notifications/unread-count`
Returns the count of unread, non-archived notifications for the user's header badge.

- **Response (200 OK)**:
```json
{
  "unread_count": 3
}
```

---

### `POST /api/v1/notifications/{notification_id}/read`
Marks a specific notification as read.

- **Security (IDOR Defense)**: Verified against requesting `user_id`. Attempting to access another user's notification returns `403 Forbidden`.

---

### `POST /api/v1/notifications/mark-all-read`
Marks all active unread notifications for the authenticated user as read.

- **Response (200 OK)**:
```json
{
  "updated_count": 3
}
```

---

### `POST /api/v1/notifications/{notification_id}/archive`
Archives a notification, removing it from default views while preserving history.

---

### `GET /api/v1/notifications/preferences`
Retrieves communication settings for the authenticated user.

- **Response (200 OK)**:
```json
{
  "email_enabled": true,
  "in_app_enabled": true,
  "donation_updates": true,
  "project_updates": true,
  "milestone_updates": true,
  "verification_updates": true,
  "certificate_notifications": true,
  "security_alerts": true
}
```

---

### `PUT /api/v1/notifications/preferences`
Updates user communication preferences.

- **Security Invariant**: `security_alerts` cannot be set to `false`. Any request attempting to disable security alerts is overwritten with `security_alerts = true`.

