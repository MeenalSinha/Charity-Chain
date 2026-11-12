# CharityChain — Notification Event Catalog

## 1. Overview

CharityChain standardizes all domain events emitted across the platform under a strongly-typed catalog (`DomainEventType`). Events are encoded as immutable `DomainEvent` envelopes serialized using **RFC 8785 Canonical JSON** with deterministic attribute sorting and automatic recursive redaction of sensitive credentials.

---

## 2. Event Envelope Structure

Every event recorded to the outbox adheres to the canonical schema:

```json
{
  "id": "evt_0a1b2c3d4e5f6789",
  "type": "payment.captured",
  "version": "v1.0.0",
  "created_at": "2026-11-12T22:30:00.000000Z",
  "entity_type": "payment_capture",
  "entity_id": "cap-998877",
  "organization_id": "org-112233",
  "actor_user_id": "usr-445566",
  "correlation_id": "req-778899",
  "idempotency_key": "payment.captured:cap-998877:evt_0a1b2c3d4e5f6789",
  "data": {
    "donation_id": "don-112233",
    "project_title": "Clean Water Wells",
    "amount_minor_units": 5000,
    "currency": "USD"
  }
}
```

---

## 3. Catalog of Domain Events

### User & Organization Lifecycle
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `user.registered` | `user` | Account registered on platform | In-App, Email |
| `user.email_verified` | `user` | Email confirmation verified | In-App, Email |
| `organization.registered` | `organization` | NGO onboarded on platform | In-App, Webhook |
| `organization.verified` | `organization` | Onboarding compliance approved | In-App, Email, Webhook |
| `organization.rejected` | `organization` | Onboarding rejected | In-App, Email, Webhook |
| `organization.suspended` | `organization` | NGO suspended by platform admin | In-App, Email, Webhook |

### Projects & Milestones
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `project.created` | `project` | Charitable project draft created | In-App |
| `project.published` | `project` | Project approved and published | In-App, Webhook |
| `project.approved` | `project` | Platform approval granted | In-App, Email, Webhook |
| `project.suspended` | `project` | Project paused / suspended | In-App, Email, Webhook |
| `milestone.created` | `milestone` | Project milestone declared | In-App |
| `milestone.completed` | `milestone` | Milestone verified & finished | In-App, Email, Webhook |

### Donations & Financial Receipts
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `donation.created` | `donation` | Donor donation intent initiated | In-App |
| `payment.authorized` | `payment_order` | Gateway authorization created | In-App |
| `payment.captured` | `payment_capture` | Approved PayPal capture executed | In-App, Email, Webhook |
| `payment.failed` | `payment_order` | Gateway declined donation | In-App, Email |
| `payment.refunded` | `payment_refund` | Full or partial refund issued | In-App, Email, Webhook |
| `payment.disputed` | `payment_dispute`| Chargeback or dispute opened | In-App, Email, Webhook |

### Evidence & AI Verification
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `evidence.submitted` | `evidence_file` | NGO uploaded milestone evidence | In-App, Webhook |
| `evidence.processing_started` | `evidence_job` | Forensic processing queued | In-App |
| `ai_verification.completed` | `verification_report`| YOLO, pHash & ELA completed | In-App, Webhook |
| `ai_verification.failed` | `verification_report`| Forensic engine error | In-App, Email |
| `evidence.additional_requested`| `review_case` | Reviewer requested revision | In-App, Email, Webhook |

### Human Review & Governance
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `review.assigned` | `review_assignment`| Reviewer assigned to case | In-App, Email |
| `review.approved` | `review_case` | Independent review approved | In-App, Webhook |
| `review.rejected` | `review_case` | Independent review rejected | In-App, Email, Webhook |
| `review.dual_key_completed` | `review_case` | Both independent keys approved | In-App, Email, Webhook |
| `dispute.opened` | `dispute_case` | Milestone dispute raised | In-App, Email, Webhook |
| `dispute.resolved` | `dispute_case` | Governance committee resolved | In-App, Email, Webhook |

### Fund Disbursements & Ledger
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `disbursement.requested` | `disbursement` | Operator requested milestone payout | In-App |
| `disbursement.approved` | `disbursement` | Checker approved maker request | In-App, Webhook |
| `disbursement.processing` | `payout_batch` | PayPal Payouts batch submitted | In-App |
| `disbursement.completed` | `disbursement` | PayPal webhook confirmed payout | In-App, Email, Webhook |
| `disbursement.failed` | `disbursement` | Payout failed or returned | In-App, Email, Webhook |

### Impact Certificates & Attestations
| Event Type | Entity Type | Trigger Description | Default Channels |
|---|---|---|---|
| `certificate.issued` | `impact_certificate`| Ed25519 digital certificate minted | In-App, Email, Webhook |
| `certificate.suspended`| `impact_certificate`| Certificate held pending review | In-App, Email, Webhook |
| `certificate.revoked` | `impact_certificate`| Certificate revoked on ledger | In-App, Email, Webhook |
| `certificate.superseded`| `impact_certificate`| Replaced by updated certificate | In-App, Webhook |
| `impact_ledger.checkpoint_created`| `checkpoint`| Periodic Merkle checkpoint | Webhook |

