# CharityChain — Public Certificate & Impact Ledger API Reference
## OpenAPI Specification, Verification Contracts, and Cryptographic Proof Endpoints

---

## 1. Overview & Base Paths

CharityChain provides open, unauthenticated public verification endpoints allowing any donor, auditor, or external system to verify digital impact certificates and cryptographic ledger integrity without requiring an API token. In parallel, authorized administrative endpoints allow reviewers and platform administrators to govern certificate lifecycles.

- **Public Base Path**: `/api/v1/public/certificates`
- **JWKS Discovery Paths**: `/.well-known/jwks.json` and `/api/v1/.well-known/jwks.json`
- **Impact Ledger Base Path**: `/api/v1/impact-ledger`
- **Certificate Administration Path**: `/api/v1/certificates`

---

## 2. Public Verification Endpoints (Unauthenticated)

### 2.1 Public Certificate Verification
- **Method & Path**: `GET /api/v1/public/certificates/{verification_id}`
- **Description**: Retrieves sanitized certificate details, current status, and cryptographic signature validity. This endpoint powers the QR code scan destination.
- **Rate Limit**: 60 requests / minute / IP.
- **Path Parameter**:
  - `verification_id` (string, required): Deterministic high-entropy verification slug.

#### Response `200 OK`
```json
{
  "valid": true,
  "status": "ISSUED",
  "verification_id": "v_f9b3e1a04d2c8",
  "certificate_number": "CC-CERT-2026-000142",
  "certificate_type": "CONTRIBUTION",
  "issued_at": "2026-11-12T14:30:00Z",
  "organization_name": "Clean Water Initiative",
  "project_title": "Rift Valley Well Installation",
  "milestone_title": "Phase 1: Hydrological Survey & Well Drilling",
  "donor": {
    "display_name": "Anonymous Contributor",
    "is_anonymous": true
  },
  "impact_summary": "1,200 villagers provided with certified potable water access.",
  "metrics": {
    "wells_drilled": 3,
    "flow_rate_lpm": 85
  },
  "cryptographic_proof": {
    "algorithm": "Ed25519",
    "key_id": "key_2026_primary",
    "canonicalization": "RFC-8785",
    "signature_b64": "vA7G4...==",
    "claims_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "signature_valid": true,
    "jwks_uri": "https://api.charitychain.org/.well-known/jwks.json"
  }
}
```

#### Error Responses
- `404 Not Found`:
  ```json
  { "detail": "Certificate not found or verification ID invalid" }
  ```
- `422 Unprocessable Entity`: Verification ID malformed.

---

### 2.2 Cryptographic Proof Envelope
- **Method & Path**: `GET /api/v1/public/certificates/{verification_id}/proof`
- **Description**: Returns the raw canonical claims payload, the cryptographic signature, and the issuer public key so that third-party verifiers can validate the signature completely offline without trusting CharityChain backend servers.
- **Path Parameter**:
  - `verification_id` (string, required)

#### Response `200 OK`
```json
{
  "verification_id": "v_f9b3e1a04d2c8",
  "certificate_number": "CC-CERT-2026-000142",
  "algorithm": "Ed25519",
  "key_id": "key_2026_primary",
  "canonical_claims_b64": "eyJhZGRpdGlvbmFsX21ldHJpY3MiOnsid2VsbHN..."",
  "claims_sha256": "8f434346648f6b96df89dda901c5176b10f60753b8c012840a7ff8a56d78883e",
  "signature_b64": "G6Z8z+v2N8p...==",
  "public_key_b64": "XbXqR...==",
  "canonical_json_spec": "RFC 8785",
  "offline_instructions": "Decode canonical_claims_b64, compute SHA-256 or verify Ed25519 signature directly against public_key_b64."
}
```

---

### 2.3 Download Verifiable PDF
- **Method & Path**: `GET /api/v1/public/certificates/{verification_id}/download`
- **Description**: Streams the rendered landscape A4 PDF certificate containing the secure verification QR code and cryptographic footer.
- **Headers**:
  - `Content-Type: application/pdf`
  - `Content-Disposition: attachment; filename="CharityChain_Certificate_CC-CERT-2026-000142.pdf"`

---

### 2.4 RFC 7517 JWKS Public Key Discovery
- **Method & Path**: `GET /.well-known/jwks.json` and `GET /api/v1/.well-known/jwks.json`
- **Description**: Standard JSON Web Key Set exposing all active and historical Ed25519 public keys used for certificate and checkpoint signing.

#### Response `200 OK`
```json
{
  "keys": [
    {
      "kty": "OKP",
      "crv": "Ed25519",
      "use": "sig",
      "alg": "EdDSA",
      "kid": "key_2026_primary",
      "x": "O2aFRL2STxPNe...==",
      "status": "ACTIVE"
    },
    {
      "kty": "OKP",
      "crv": "Ed25519",
      "use": "sig",
      "alg": "EdDSA",
      "kid": "key_2025_retired",
      "x": "u43nfdks903...==",
      "status": "RETIRED"
    }
  ]
}
```

---

## 3. Impact Ledger Public Verification Endpoints

### 3.1 Verify Milestone Event Chain
- **Method & Path**: `GET /api/v1/impact-ledger/milestones/{milestone_id}/verify`
- **Description**: Performs a real-time cryptographic integrity audit across all events in the milestone's ledger stream, verifying the Genesis block, hash chain links ($H_i = \text{SHA256}(H_{i-1} \parallel E_i)$), and signed checkpoint tree.

#### Response `200 OK`
```json
{
  "stream_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "milestone_id": "7ca64e52-1234-4562-b3fc-9c847a66afc2",
  "total_events": 6,
  "verified": true,
  "chain_valid": true,
  "checkpoints_verified": 1,
  "latest_sequence": 6,
  "latest_event_hash": "9c4d28f89c6298...==",
  "audit_timestamp": "2026-11-12T18:40:00Z"
}
```

---

## 4. Administrative Certificate Lifecycle Endpoints (Authenticated)

Access requires `PLATFORM_ADMIN` or authorized governance roles.

### 4.1 Suspend Certificate
- **Method & Path**: `POST /api/v1/certificates/{certificate_id}/suspend`
- **Payload**:
  ```json
  {
    "reason": "Temporary investigation into reported field discrepancy",
    "reason_code": "SUSPECTED_FRAUD"
  }
  ```
- **Response `200 OK`**: Updated certificate model with `status: "SUSPENDED"`. Emits `CERTIFICATE_SUSPENDED` event to ledger.

### 4.2 Revoke Certificate
- **Method & Path**: `POST /api/v1/certificates/{certificate_id}/revoke`
- **Payload**:
  ```json
  {
    "reason": "Milestone approval overturned by independent governance board review",
    "reason_code": "REVIEW_OVERTURNED"
  }
  ```
- **Response `200 OK`**: Updated certificate with `status: "REVOKED"` and recorded `revoked_at`.

### 4.3 Supersede Certificate
- **Method & Path**: `POST /api/v1/certificates/{certificate_id}/supersede`
- **Payload**:
  ```json
  {
    "replacement_claims": {
      "metrics": { "wells_drilled": 4 }
    },
    "reason": "Rectified metric count following final field audit"
  }
  ```
- **Response `201 Created`**: New certificate object referencing `supersedes_id`, while original transitions to `status: "SUPERSEDED"`.

