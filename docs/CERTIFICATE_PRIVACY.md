# CharityChain — Certificate Privacy Architecture & PII Protection
## Privacy-by-Design, Claim Segregation, and GDPR Compliance for Verifiable Credentials

---

## 1. Executive Summary & Design Principles

CharityChain's **Proof-of-Impact Certificates** provide tamper-evident, cryptographically verifiable claims that charitable contributions achieved their documented real-world milestones. However, public verifiability must never compromise donor, reviewer, or beneficiary privacy.

The platform implements **Privacy by Design and by Default** (in accordance with **GDPR Article 25**):
1. **Public vs. Private Claim Segregation**: The cryptographic payload signed by the platform issuer contains structured, verifiable assertions. When rendered publicly, claims are passed through a strict redaction pipeline.
2. **Explicit Consent for Attribution**: Donors are anonymous by default on public records (`"Anonymous Contributor"`). Personal attribution is displayed only when the donor has explicitly selected `public_attribution=True`.
3. **Absolute PII Suppression**: Highly sensitive personally identifiable information (donor email addresses, phone numbers, payment instrument identifiers, internal review rationale) is **never** emitted on public APIs, QR scan outputs, or public verification landing pages, even if attribution consent is granted.
4. **Beneficiary Protection**: Sensitive beneficiary identifying details (e.g., vulnerable populations, medical records) are strictly excluded from on-chain and off-chain certificate claims.

---

## 2. Public vs. Private Claim Segregation Matrix

Every certificate contains a complete internal claims envelope (`claims_data`) used for cryptographic hashing and auditing, and a sanitized public view (`public_claims`) returned by public endpoints (`/api/v1/public/certificates/{verification_id}`).

| Claim Element | Internal Signed Envelope (`claims_data`) | Public Verification Endpoint (`GET /api/v1/public/certificates/{id}`) | Public PDF / QR Code Output |
| :--- | :--- | :--- | :--- |
| **Certificate ID / Number** | Included (UUID & Human-Readable Code) | Included | Included |
| **Verification ID** | Included (Deterministic URL slug) | Included | Included (in QR URL) |
| **Certificate Type** | Included (`CONTRIBUTION`, `MILESTONE_COMPLETION`) | Included | Included |
| **Issuance Timestamp** | Included (ISO-8601 UTC) | Included | Included |
| **Organization Name** | Included | Included | Included |
| **Project & Milestone Name**| Included | Included | Included |
| **Milestone Impact Summary** | Included (Public impact metrics) | Included | Included |
| **Donor Identifier (`donor_id`)** | Included (UUID) | Redacted (`null`) | Redacted (`null`) |
| **Donor Email** | Excluded / Suppressed | **Strictly Excluded (Suppressed)** | **Strictly Excluded (Suppressed)** |
| **Donor Name / Attribution** | Stored / Preserved | Conditional: Display Name if `public_attribution=True`, else `"Anonymous Contributor"` | Conditional: Display Name if `public_attribution=True`, else `"Anonymous Contributor"` |
| **Financial Order & Capture IDs**| Stored in internal audit link | **Strictly Redacted** | **Strictly Redacted** |
| **Contribution Amount** | Stored in contribution claims | Displayed formatted (e.g., `USD 100.00`) | Displayed formatted |
| **Reviewer Personal Identities**| Excluded from certificate | **Strictly Excluded** (Only reviewer count & governance status shown) | **Strictly Excluded** |
| **Reviewer Internal Notes** | Excluded from certificate | **Strictly Excluded** | **Strictly Excluded** |
| **Cryptographic Signature** | Included (Base64 Ed25519) | Included in verification proof | Embedded in QR code |
| **Issuer Key ID (`kid`)** | Included | Included | Included |

---

## 3. Donor Attribution & Anonymity Lifecycle

### 3.1 Consent Capture
When an individual donor initiates a contribution or configures their profile:
- `public_attribution` defaults to `False`.
- The user may opt-in by selecting "Display my name on public impact certificate and transparency records".

### 3.2 Public Sanitization Logic
In `backend.app.services.public_verification_service.PublicVerificationService`:
```python
def _sanitize_public_claims(self, cert: ImpactCertificate) -> Dict[str, Any]:
    raw_claims = cert.claims_data or {}
    
    # Strip all internal identifiers and payment instrumentation
    sanitized = {
        "certificate_id": str(cert.id),
        "certificate_number": cert.certificate_number,
        "verification_id": cert.verification_id,
        "certificate_type": cert.certificate_type.value,
        "status": cert.status.value,
        "issued_at": cert.issued_at.isoformat() if cert.issued_at else None,
        "organization_name": cert.recipient_name if cert.certificate_type == CertificateType.MILESTONE_COMPLETION else raw_claims.get("organization_name"),
        "project_title": raw_claims.get("project_title"),
        "milestone_title": raw_claims.get("milestone_title"),
        "impact_summary": raw_claims.get("impact_summary"),
        "metrics": raw_claims.get("metrics", {}),
    }

    # Donor Privacy Redaction
    if cert.certificate_type == CertificateType.CONTRIBUTION:
        if cert.public_attribution:
            sanitized["donor"] = {
                "display_name": cert.recipient_name or raw_claims.get("donor_name", "Supporter"),
                "is_anonymous": False
            }
        else:
            sanitized["donor"] = {
                "display_name": "Anonymous Contributor",
                "is_anonymous": True
            }
        # Absolute invariant: email is never present in sanitized public payload
        if "email" in sanitized.get("donor", {}):
            del sanitized["donor"]["email"]

    return sanitized
```

### 3.3 Protection Against Correlation & Enumeration
- Public verification utilizes a high-entropy `verification_id` (cryptographically secure alphanumeric token) rather than predictable sequential IDs or donor UUIDs.
- Sequential iteration attacks cannot discover donor contribution history or link multiple certificates to a single donor.

---

## 4. GDPR & Regulatory Compliance Architecture

### 4.1 GDPR Article 17 ("Right to Erasure" / Right to be Forgotten)
Because digital certificates and impact ledgers contain hash-chained attestations, data mutability must be handled deliberately:
- **No Direct PII in Signatures**: The canonical JSON claims signed by the Ed25519 private key do **not** include unhashed government identity numbers, physical home addresses, or direct donor email addresses.
- **Revocation and Redaction**: If a donor requests erasure:
  1. The platform issues a `CertificateRevocation` or `CertificateSupersession`.
  2. The certificate status transitions to `REVOKED` or `SUPERSEDED`.
  3. The `recipient_name` is scrubbed and set to `"Anonymous Contributor"` in the database.
  4. Public endpoints immediately reflect `"Anonymous Contributor"` and revocation metadata.
  5. The cryptographic ledger preserves the chronological record that a certificate was revoked without persisting the deleted name.

### 4.2 GDPR Article 25 ("Data Protection by Design and by Default")
- Default setting is privacy-preserving (`public_attribution = False`).
- Minimization: Only data points necessary to verify milestone achievement and financial contribution existence are processed.

### 4.3 Reviewer Independence & Identity Shielding
- Reviewers perform critical governance functions (evaluating evidence, rejecting fraudulent claims, demanding revisions).
- Exposing individual reviewer names or contact details could expose reviewers to coercion, harassment, or bribery from rogue NGOs.
- Consequently, impact certificates attest to **systemic governance compliance** (`dual_key_approved: true`, `reviewer_count: 2`) while shielding reviewer personal credentials.

---

## 5. Security & Verification Testing

The privacy framework is verified by automated test suites:
- `tests/unit/test_certificate_privacy.py`:
  - `test_default_anonymity()`: Verifies that certificates issued without attribution show `"Anonymous Contributor"` and `is_anonymous: True`.
  - `test_email_exclusion()`: Validates that donor email addresses are never returned in public verification API responses, even when `public_attribution=True`.
- `tests/security/test_certificate_idor.py`:
  - Validates that non-admin and non-owner callers cannot retrieve full unredacted internal certificate models.

