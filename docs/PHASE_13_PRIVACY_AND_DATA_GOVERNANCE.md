# CharityChain — Phase 13 Privacy and Data Governance Policy
**Document Version:** 1.0.0  
**Scope:** Data Classification, Beneficiary Protection, Donor Anonymity, Retention Schedules, and AI Governance  

---

## 1. Data Classification Matrix

| Classification Level | Definition | Examples | Handling Requirements |
|---|---|---|---|
| **Restricted (Tier 1)** | Catastrophic impact if exposed | Private Ed25519 signing keys, JWT secrets, database credentials | Encrypted at rest, never logged, KMS or env protected |
| **Confidential (Tier 2)** | High impact to individuals or NGOs | NGO tax IDs, donor payment records, user passwords, private review notes | Argon2id hashed, encrypted in transit, RBAC enforced |
| **Internal (Tier 3)** | Operational data for internal staff | Unpublished milestone drafts, AI forensic intermediate scores | Access restricted to authenticated users |
| **Public (Tier 4)** | Data intended for open public audit | Public project descriptions, verified certificates, global transparency KPIs | Freely accessible via `/api/v1/transparency` |

---

## 2. Beneficiary Protection & Location Privacy

### 2.1 The Beneficiary Harm Risk
Certain humanitarian aid projects (e.g., domestic violence shelters, refugee relief, endangered wildlife sanctuaries) could endanger vulnerable individuals if their exact physical coordinates are publicly exposed.

### 2.2 Centroid Coarsening Algorithm
When a project or milestone is flagged as `is_location_sensitive = True`:
- Exact GPS coordinates extracted from EXIF metadata are stripped from all public API responses.
- Coordinates are coarsened to the regional centroid (~0.1 decimal degree grid, approximately 11.1 km radius).
- The public map displays a bounded geographic circle rather than a pinpoint marker.

---

## 3. Donor Anonymity & Identity Protection

- **Public Display**: Donors have the right to contribute anonymously. On public certificate verification and public donor lists, anonymous donations are rendered as `"Anonymous Contributor"`.
- **Payment Token Segregation**: PayPal payment tokens, card numbers, and banking details are never stored on CharityChain servers. All transaction references use PayPal order/capture IDs.
- **Certificate Proof Envelopes**: Proof-of-Impact certificates cryptographically commit to the contribution without revealing the donor's email address or full name unless explicitly opted in.

---

## 4. Data Retention & Secure Disposal

| Data Category | Retention Period | Disposal Mechanism |
|---|:---:|---|
| **Financial Ledger Records** | 7 Years (Legal requirement) | Archived to immutable cold storage |
| **Security Audit Logs** | 1 Year (365 days) | Automatic rotation or archived export |
| **User Sessions** | 7 Days | Expired records purged from database |
| **Verification Tokens** | 24 Hours | Expired single-use tokens deleted |
| **Raw Evidence Media** | Project Duration + 3 Years | Cryptographic erasure of local/cloud blobs |

---

## 5. AI Ethics & Verification Governance

1. **Human-in-the-Loop Principle**: AI models (YOLO object detection, Error Level Analysis, pHash comparison) never make binding financial or approval decisions autonomously. AI serves exclusively as an advisory diagnostic tool for human reviewers.
2. **Dual-Key Human Approval**: Releasing donor funds requires two independent human reviewers to approve the evidence packet.
3. **Forensic Transparency**: All AI confidence scores, detected objects, and ELA deviation maps are preserved in the review case for auditability.

