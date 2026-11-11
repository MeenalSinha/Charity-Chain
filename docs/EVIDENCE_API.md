# Evidence Management REST API Reference

## 1. Overview

Base Path: `/api/v1`  
Authentication: Bearer Token (JWT in `Authorization` header) or Active Session.  
Standard Envelope: All JSON responses use `StandardResponse[T]` with `success`, `data`, `message`, `timestamp`, and `correlation_id`.

---

## 2. Endpoints Summary

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/milestones/{id}/evidence` | Upload batch of evidence files | NGO Admin / Staff |
| `GET` | `/milestones/{id}/evidence` | List all evidence for a milestone | Authenticated Users |
| `GET` | `/evidence/{id}` | Get metadata for a specific evidence asset | Authenticated Users |
| `GET` | `/evidence/{id}/download` | Get presigned or internal download URL | Authorized Reviewer/Staff |
| `GET` | `/evidence/{id}/content` | Stream raw binary content | Authorized Reviewer/Staff |
| `POST` | `/evidence/{id}/versions` | Upload replacement version | NGO Admin / Staff |
| `GET` | `/evidence/{id}/versions` | Get full version lineage history | Authenticated Users |
| `POST` | `/evidence/{id}/verify-integrity` | Trigger cryptographic SHA-256 verification | Authenticated Users |
| `POST` | `/evidence/{id}/archive` | Soft-archive evidence asset | NGO Admin / Platform Admin |
| `GET` | `/evidence/{id}/audit-trail` | Get tamper-evident audit log & proof | Authenticated Users |

---

## 3. Endpoint Details

### 3.1 Multi-File Evidence Upload
`POST /api/v1/milestones/{milestone_id}/evidence`  
Content-Type: `multipart/form-data`

#### Parameters
- `milestone_id` (path, string, UUID): Target milestone identifier.
- `files` (form body, array of files): Batch of files (max: 10 files, max: 10 MB each, types: JPEG, PNG, PDF).

#### Success Response (`201 Created`)
```json
{
  "success": true,
  "data": {
    "milestone_id": "7b3b6f00-...",
    "total_uploaded": 2,
    "items": [
      {
        "id": "e4419ad2-...",
        "milestone_id": "7b3b6f00-...",
        "uploader_id": "c1f7b53a-...",
        "original_filename": "foundation_pour.jpg",
        "storage_key": "milestones/7b3b6f00-.../a1f8c..._foundation_pour.jpg",
        "storage_provider": "local",
        "file_path": "/api/v1/evidence/milestones/.../download",
        "file_hash": "a1f8c6d4e...",
        "content_address": "sha256:a1f8c6d4e...",
        "mime_type": "image/jpeg",
        "file_size_bytes": 1048576,
        "version_number": 1,
        "parent_evidence_id": null,
        "status": "UPLOADED",
        "is_archived": false,
        "archived_at": null,
        "quarantine_reason": null,
        "created_at": "2026-11-11T12:00:00Z",
        "download_url": "/api/v1/evidence/milestones/.../download"
      }
    ]
  },
  "message": "Successfully uploaded 2 evidence file(s)."
}
```

---

### 3.2 List Milestone Evidence
`GET /api/v1/milestones/{milestone_id}/evidence?include_archived=false`

#### Parameters
- `milestone_id` (path, string, UUID): Milestone identifier.
- `include_archived` (query, boolean, default: `false`): Include soft-archived records.

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "data": [
    {
      "id": "e4419ad2-...",
      "milestone_id": "7b3b6f00-...",
      "original_filename": "foundation_pour.jpg",
      "version_number": 1,
      "status": "UPLOADED",
      "file_hash": "a1f8c6d4e...",
      "is_archived": false
    }
  ]
}
```

---

### 3.3 Download Evidence Access URL
`GET /api/v1/evidence/{evidence_id}/download`

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "data": {
    "evidence_id": "e4419ad2-...",
    "filename": "foundation_pour.jpg",
    "mime_type": "image/jpeg",
    "download_url": "https://charitychain-evidence.s3.amazonaws.com/milestones/...?X-Amz-Signature=..."
  }
}
```

---

### 3.4 Upload Replacement Version
`POST /api/v1/evidence/{evidence_id}/versions`  
Content-Type: `multipart/form-data`

#### Parameters
- `evidence_id` (path, string, UUID): Parent evidence asset identifier.
- `file` (form body, file): Replacement file payload.
- `reason` (form body, string, optional): Reason for update.

#### Success Response (`201 Created`)
```json
{
  "success": true,
  "data": {
    "id": "993a4b01-...",
    "version_number": 2,
    "parent_evidence_id": "e4419ad2-...",
    "original_filename": "foundation_pour_hd.jpg",
    "file_hash": "b2c7e1f4...",
    "status": "UPLOADED"
  },
  "message": "Created version 2 successfully."
}
```

---

### 3.5 Live Cryptographic Integrity Verification
`POST /api/v1/evidence/{evidence_id}/verify-integrity`

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "data": {
    "evidence_id": "e4419ad2-...",
    "expected_hash": "a1f8c6d4e...",
    "computed_hash": "a1f8c6d4e...",
    "is_valid": true,
    "checked_at": "2026-11-11T12:05:00Z",
    "storage_provider": "local"
  },
  "message": "Integrity check completed."
}
```

---

### 3.6 Soft-Archive Evidence
`POST /api/v1/evidence/{evidence_id}/archive`

#### Request Body
```json
{
  "reason": "Superseded by certified engineering survey"
}
```

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "data": {
    "id": "e4419ad2-...",
    "is_archived": true,
    "archived_at": "2026-11-11T12:10:00Z"
  },
  "message": "Evidence asset soft-archived."
}
```

---

### 3.7 Tamper-Evident Audit Trail
`GET /api/v1/evidence/{evidence_id}/audit-trail`

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "data": {
    "evidence_id": "e4419ad2-...",
    "is_chain_intact": true,
    "total_events": 2,
    "events": [
      {
        "id": "881a7d3c-...",
        "evidence_id": "e4419ad2-...",
        "actor_id": "c1f7b53a-...",
        "event_type": "UPLOADED",
        "details_json": {
          "action": "UPLOAD",
          "original_filename": "foundation_pour.jpg",
          "file_size_bytes": 1048576,
          "timestamp": "2026-11-11T12:00:00Z"
        },
        "previous_event_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        "current_event_hash": "f4e190b2c...",
        "created_at": "2026-11-11T12:00:00Z"
      },
      {
        "id": "992b8e4d-...",
        "evidence_id": "e4419ad2-...",
        "actor_id": "c1f7b53a-...",
        "event_type": "INTEGRITY_VERIFIED",
        "details_json": {
          "is_valid": true,
          "expected_hash": "a1f8c6d4e...",
          "computed_hash": "a1f8c6d4e...",
          "timestamp": "2026-11-11T12:05:00Z"
        },
        "previous_event_hash": "f4e190b2c...",
        "current_event_hash": "c7a8b3e9f...",
        "created_at": "2026-11-11T12:05:00Z"
      }
    ]
  },
  "message": "Audit chain verified intact."
}
```
