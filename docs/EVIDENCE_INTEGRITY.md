# Evidence Integrity & Cryptographic Ledger Specification

## 1. Principles of Tamper-Evident Records

CharityChain uses cryptographic content addressing and sequential hash chaining to guarantee the integrity and historical truth of evidence records:

1. **Content Addressability**: Assets are identified by their cryptographic SHA-256 digest (`sha256:<hash>`).
2. **Deduplication**: Redundant files cannot be submitted to the same milestone.
3. **Sequential Provability**: The full audit history of an evidence asset forms a mathematically verifiable, ordered sequence of SHA-256 hashes.
4. **Instant Tamper Detection**: If any file byte is corrupted or modified on disk or cloud storage, the system immediately flags the asset as tampered.

---

## 2. Cryptographic Deduplication

When an NGO submits a file for a milestone:
1. The server reads the file stream and computes its SHA-256 checksum:
   $$\text{Hash} = \text{SHA-256}(\text{file\_bytes})$$
2. The database is queried for an active (non-archived) record on the same milestone with matching hash:
   ```sql
   SELECT id FROM evidence_files
   WHERE milestone_id = :milestone_id
     AND file_hash = :hash
     AND is_archived = false
   LIMIT 1;
   ```
3. If found, `ConflictException` (HTTP 409) is returned:
   ```json
   {
     "error": {
       "code": "CONFLICT",
       "message": "Duplicate evidence detected: File with identical SHA-256 hash already exists on milestone."
     }
   }
   ```
4. This eliminates accidental double-submissions and deliberate replay attempts.

---

## 3. Cryptographic Audit Hash Chain

Every evidence asset possesses an immutable audit chain recorded in `evidence_audit_events`.

### Mathematical Chain Formulation

- **Genesis Event ($E_0$)**:
  When a file is first uploaded:
  $$P_0 = \text{"0"}^{64} \quad (\text{64 zeros})$$
  $$H_0 = \text{SHA-256}\Big(P_0 \;\big|\; \text{evidence\_id} \;\big|\; \text{actor\_id} \;\big|\; \text{event\_type} \;\big|\; \text{canonical\_json}(\text{details})\Big)$$

- **Subsequent Event ($E_k$, for $k \ge 1$)**:
  $$P_k = H_{k-1}$$
  $$H_k = \text{SHA-256}\Big(P_k \;\big|\; \text{evidence\_id} \;\big|\; \text{actor\_id} \;\big|\; \text{event\_type} \;\big|\; \text{canonical\_json}(\text{details})\Big)$$

### Canonical JSON Representation
To ensure cross-platform reproducibility, `canonical_json` is serialized with sorted keys and compact separators:
```python
json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
```

### Supported Event Types
1. `UPLOADED`: Genesis upload record.
2. `INTEGRITY_VERIFIED`: Successful cryptographic content verification.
3. `INTEGRITY_TAMPERED`: Failed cryptographic content verification.
4. `VERSION_CREATED`: Submission of a successor version.
5. `SUPERSEDED`: Historical asset superseded by a new version.
6. `DOWNLOADED`: Asset downloaded by an authorized reviewer or staff member.
7. `ARCHIVED`: Asset soft-archived by authorized staff.

---

## 4. Verification Algorithm (`get_audit_trail`)

Anyone authorized can verify the mathematical continuity of an evidence ledger:

```python
is_chain_intact = True
expected_prev_hash = None

for idx, ev in enumerate(events):
    # 1. Verify chain link continuity
    if idx > 0 and ev.previous_event_hash != expected_prev_hash:
        is_chain_intact = False
        break

    # 2. Recompute deterministic SHA-256 hash
    computed = compute_audit_hash(
        previous_hash=ev.previous_event_hash,
        evidence_id=ev.evidence_id,
        actor_id=ev.actor_id,
        event_type=ev.event_type,
        details=ev.details_json
    )

    # 3. Verify match against recorded hash
    if computed != ev.current_event_hash:
        is_chain_intact = False
        break

    expected_prev_hash = ev.current_event_hash
```

If any log record is deleted, inserted out of order, or edited in the database, `is_chain_intact` evaluates to `False`.

---

## 5. Live File Integrity Checking

The `/api/v1/evidence/{id}/verify-integrity` endpoint performs real-time storage auditing:
1. Fetches raw bytes directly from the active storage backend (`LocalStorageProvider` or `S3StorageProvider`).
2. Calculates current SHA-256 of the fetched bytes.
3. Compares against `EvidenceFile.file_hash`.
4. If identical: returns `is_valid: true` and logs `INTEGRITY_VERIFIED`.
5. If mismatched (e.g. disk corruption, unauthorized replacement): returns `is_valid: false` and logs `INTEGRITY_TAMPERED`.

