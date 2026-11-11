# CharityChain — Human Review & Decision Workflows API Specification

## 1. Overview

The Human Review REST API (`/api/v1/reviews/*`) provides endpoints for managing review cases, reviewer queues, dual-key decision submissions, split adjudications, revision rounds, formal disputes, and cryptographic audit trails.

All endpoints require authentication via JWT Bearer token and enforce role-based access control (RBAC).

---

## 2. Review Case Management Endpoints

### 2.1 Submit Milestone for Review
- **Route**: `POST /api/v1/reviews/cases/milestone/{milestone_id}/submit`
- **Role**: `ORGANIZATION_ADMIN`, `ORGANIZATION_OWNER`, `ORGANIZATION_MEMBER`, `PLATFORM_ADMIN`
- **Request Body**:
  ```json
  {
    "submission_notes": "Foundation completed with certified engineering report."
  }
  ```
- **Response**: `201 Created` — `ReviewCaseResponse`

### 2.2 List Review Cases
- **Route**: `GET /api/v1/reviews/cases`
- **Role**: Any authenticated user (scoped to user's organization for NGOs; all cases for Admins/Reviewers)
- **Query Parameters**:
  - `status_filter` (optional): `SUBMITTED`, `ASSIGNED`, `IN_REVIEW`, `AWAITING_SECOND_REVIEW`, `APPROVED`, `REVISION_REQUIRED`, `REJECTED`, `ESCALATED`, `DISPUTED`, `CLOSED`
  - `organization_id` (optional): UUID
- **Response**: `200 OK` — `List[ReviewCaseResponse]`

### 2.3 Get Review Case Detail (Blind Review Masked)
- **Route**: `GET /api/v1/reviews/cases/{case_id}`
- **Role**: Assigned Reviewers, Platform Admins, NGO Members of Organization
- **Response**: `200 OK` — Complete case payload. If caller is an assigned reviewer whose decision is still pending, other reviewers' decisions are masked as `REDACTED_BLIND_REVIEW` and `SUBMITTED_HIDDEN`.

### 2.4 Assign Dual Reviewers
- **Route**: `POST /api/v1/reviews/cases/{case_id}/assign`
- **Role**: `PLATFORM_ADMIN`
- **Request Body**:
  ```json
  {
    "reviewer_1_id": "uuid-reviewer-1",
    "reviewer_2_id": "uuid-reviewer-2"
  }
  ```
- **Response**: `201 Created` — `List[ReviewTaskResponse]`

---

## 3. Reviewer Queue & Task Endpoints

### 3.1 Get Reviewer Task Queue
- **Route**: `GET /api/v1/reviews/queue`
- **Role**: `REVIEWER`, `PLATFORM_ADMIN`
- **Query Parameters**: `status_filter` (`ASSIGNED`, `ACCEPTED`, `IN_PROGRESS`, `DECIDED`, `DECLINED`)
- **Response**: `200 OK` — List of assigned tasks enriched with project and milestone metadata.

### 3.2 Accept Assignment
- **Route**: `POST /api/v1/reviews/tasks/{task_id}/accept`
- **Role**: Assigned Reviewer or Platform Admin
- **Response**: `200 OK` — `ReviewTaskResponse` (`assignment_state = "ACCEPTED"`)

### 3.3 Decline Assignment
- **Route**: `POST /api/v1/reviews/tasks/{task_id}/decline`
- **Role**: Assigned Reviewer or Platform Admin
- **Request Body**:
  ```json
  {
    "reason": "Expertise mismatch in deep well geothermal drilling."
  }
  ```
- **Response**: `200 OK` — `ReviewTaskResponse` (`assignment_state = "DECLINED"`)

### 3.4 Declare Conflict of Interest
- **Route**: `POST /api/v1/reviews/tasks/{task_id}/conflict`
- **Role**: Assigned Reviewer or Platform Admin
- **Request Body**:
  ```json
  {
    "reason": "Discovered that the contractor is a close relative."
  }
  ```
- **Response**: `200 OK` — `ReviewTaskResponse`

---

## 4. Decision & Adjudication Endpoints

### 4.1 Submit Review Decision
- **Route**: `POST /api/v1/reviews/tasks/{task_id}/decision`
- **Role**: Assigned Reviewer or Platform Admin
- **Request Body**:
  ```json
  {
    "decision": "APPROVE",
    "comments": "Concrete core samples verify compressive strength exceeds 30 MPa specifications.",
    "structured_revision_data": null,
    "rejection_reason_category": null
  }
  ```
- **Response**: `201 Created` — `ReviewDecisionResponse`

### 4.2 Adjudicate Escalated Case
- **Route**: `POST /api/v1/reviews/cases/{case_id}/adjudicate`
- **Role**: `PLATFORM_ADMIN`
- **Request Body**:
  ```json
  {
    "decision": "APPROVE",
    "comments": "After independent senior structural evaluation, the GPS waypoint discrepancy is attributed to consumer satellite jitter. Physical ground evidence confirms accurate placement."
  }
  ```
- **Response**: `200 OK` — `ReviewCaseResponse`

---

## 5. Revision & Resubmission Endpoints

### 5.1 Submit Revision Response
- **Route**: `POST /api/v1/reviews/cases/{case_id}/revision-response`
- **Role**: Organization Admins / Members
- **Request Body**:
  ```json
  {
    "written_response": "We have uploaded high-definition drone images and surveyor certified boundary measurements.",
    "new_evidence_file_ids": ["uuid-evidence-1", "uuid-evidence-2"]
  }
  ```
- **Response**: `201 Created` — `ReviewCaseResponse` (Round N+1)

---

## 6. Dispute & Appeal Endpoints

### 6.1 Raise Formal Dispute
- **Route**: `POST /api/v1/reviews/cases/{case_id}/disputes`
- **Role**: Organization Admins
- **Request Body**:
  ```json
  {
    "dispute_category": "EVIDENCE_MISINTERPRETATION",
    "dispute_reason": "Reviewer interpreted dark roof patch as incomplete solar panels, whereas thermal imaging confirms these are high-efficiency monocrystalline panels.",
    "supporting_evidence": {"thermal_analysis_pdf": "hash123"}
  }
  ```
- **Response**: `201 Created` — `ReviewDisputeResponse`

### 6.2 Get Dispute Detail
- **Route**: `GET /api/v1/reviews/disputes/{dispute_id}`
- **Role**: Organization Members, Assigned Adjudicator, Platform Admin
- **Response**: `200 OK` — Dispute object with threaded discussion messages.

### 6.3 Add Dispute Message
- **Route**: `POST /api/v1/reviews/disputes/{dispute_id}/messages`
- **Role**: Dispute Participants
- **Request Body**:
  ```json
  {
    "message": "Manufacturer warranty and serial numbers have been uploaded."
  }
  ```
- **Response**: `201 Created` — Dispute message payload.

### 6.4 Resolve Dispute
- **Route**: `POST /api/v1/reviews/disputes/{dispute_id}/resolve`
- **Role**: `PLATFORM_ADMIN` (independent from original reviewers)
- **Request Body**:
  ```json
  {
    "resolution": "OVERTURN_ORIGINAL_DECISION",
    "explanation": "Manufacturer certification and thermal telemetry confirm proper completion. Rejection is overturned."
  }
  ```
- **Response**: `200 OK` — `ReviewDisputeResponse`

---

## 7. Audit Trail Verification Endpoint

### 7.1 Verify Milestone Review Audit Trail
- **Route**: `GET /api/v1/reviews/audit/milestone/{milestone_id}/verify`
- **Role**: Any authenticated user
- **Response**: `200 OK` —
  ```json
  {
    "milestone_id": "uuid-milestone",
    "total_events": 7,
    "is_chain_intact": true,
    "events": [...]
  }
  ```
