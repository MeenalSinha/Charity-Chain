# AI Verification REST API Specification — CharityChain Phase 6

## 1. Overview

The AI Verification API provides endpoints for triggering verification jobs, monitoring background processing, retrieving structured verification reports, inspecting historical audit runs, and downloading diagnostic forensic artifacts.

- **Base URL**: `/api/v1/verification`
- **Authentication**: Bearer JWT (`Authorization: Bearer <token>`)
- **Access Control**: Role-based access control (RBAC) enforced per endpoint.

---

## 2. Endpoints Summary

| Method | Endpoint | Allowed Roles | Description |
|--------|----------|---------------|-------------|
| `POST` | `/api/v1/verification/jobs` | `PLATFORM_ADMIN`, `REVIEWER`, `ORGANIZATION_OWNER`, `ORGANIZATION_ADMIN`, `ORGANIZATION_MEMBER` | Enqueue a verification job for an evidence asset. |
| `GET` | `/api/v1/verification/jobs/{job_id}` | Same as above (scoped to evidence asset) | Retrieve status and completion details of a verification job. |
| `GET` | `/api/v1/verification/reports/evidence/{evidence_id}/latest` | Same as above (scoped to evidence asset) | Retrieve the most recent verification report for an evidence asset. |
| `GET` | `/api/v1/verification/reports/{report_id}` | Same as above (scoped to evidence asset) | Retrieve detailed report with all individual check records. |
| `GET` | `/api/v1/verification/evidence/{evidence_id}/history` | Same as above (scoped to evidence asset) | Retrieve full audit history of verification reports for an asset. |
| `GET` | `/api/v1/verification/artifacts/{artifact_id}` | `PLATFORM_ADMIN`, `REVIEWER`, `ORGANIZATION_OWNER`, `ORGANIZATION_ADMIN`, `ORGANIZATION_MEMBER` | Stream and download diagnostic forensic artifact (e.g. ELA heatmap). |

---

## 3. Endpoint Details

### 3.1 Submit Verification Job
`POST /api/v1/verification/jobs`

#### Request Body
```json
{
  "evidence_file_id": "b7623de4-4191-45bb-9487-8c87044dbaf1",
  "baseline_evidence_id": "c8734ef5-5202-56cc-0598-9d98155ecbf2",
  "priority": 1
}
```

#### Response (`202 Accepted`)
```json
{
  "id": "e4912ab3-3801-4478-831e-450989adcb01",
  "evidence_file_id": "b7623de4-4191-45bb-9487-8c87044dbaf1",
  "baseline_evidence_id": "c8734ef5-5202-56cc-0598-9d98155ecbf2",
  "status": "PROCESSING",
  "attempt_count": 1,
  "max_attempts": 3,
  "report_id": "a190bce2-9012-42da-91ef-817290abcdef",
  "error_category": null,
  "error_message": null,
  "started_at": "2026-11-12T00:15:00.000000Z",
  "completed_at": "2026-11-12T00:15:01.500000Z",
  "created_at": "2026-11-12T00:15:00.000000Z"
}
```

---

### 3.2 Get Latest Verification Report
`GET /api/v1/verification/reports/evidence/{evidence_id}/latest`

#### Response (`200 OK`)
```json
{
  "id": "a190bce2-9012-42da-91ef-817290abcdef",
  "evidence_file_id": "b7623de4-4191-45bb-9487-8c87044dbaf1",
  "evidence_version": 1,
  "evidence_hash": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
  "project_id": "310467b7-5f72-474c-83b5-4b0e36ce51b1",
  "milestone_id": "c71e72e1-a068-466a-8488-8285fb70c3ba",
  "pipeline_version": "1.0.0",
  "overall_assessment": "CONSISTENT",
  "confidence_score": 0.85,
  "calibration_status": "UNVALIDATED",
  "processing_started_at": "2026-11-12T00:15:00.000000Z",
  "processing_completed_at": "2026-11-12T00:15:01.500000Z",
  "provider_name": "yolo_forensics",
  "model_metadata_json": {
    "yolo_model": "yolov8n",
    "weights_hash": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
  },
  "summary_notes": "All indicators consistent with authentic milestone completion.",
  "limitations_json": [
    "Confidence scores represent heuristic multi-indicator weights (UNVALIDATED).",
    "Object detection vocabulary is bounded by the 80 COCO classes."
  ],
  "checks": [
    {
      "check_type": "FILE_INTEGRITY",
      "status": "PASSED",
      "passed": true,
      "message": "SHA-256 hash verified against immutable asset record.",
      "duration_ms": 2.4,
      "diagnostic_artifact_path": null
    },
    {
      "check_type": "OBJECT_DETECTION",
      "status": "PASSED",
      "passed": true,
      "message": "Found 2 relevant objects matching category WATER_SANITATION.",
      "duration_ms": 145.2,
      "diagnostic_artifact_path": null
    },
    {
      "check_type": "ERROR_LEVEL_ANALYSIS",
      "status": "PASSED",
      "passed": true,
      "message": "Mean compression error level: 4.2. No significant spatial manipulation anomalies detected.",
      "duration_ms": 25.1,
      "diagnostic_artifact_path": "diagnostics/ela_a1b2c3d4.png"
    }
  ]
}
```

---

### 3.3 Get Diagnostic Artifact
`GET /api/v1/verification/artifacts/{artifact_id}`

#### Response (`200 OK`)
Streams the binary image (`image/png` or `image/jpeg`) with cache control headers.
Path traversal attempts (`../`) are detected and rejected with `400 Bad Request`.
Donors and unauthenticated clients receive `403 Forbidden`.
