# Evidence Security Architecture

## 1. Threat Model & Security Posture

Evidence assets represent high-value data on CharityChain: milestone approvals and subsequent financial releases depend directly on their authenticity. The Phase 5 security architecture addresses five critical threat categories:

1. **Malicious File Ingestion**: Upload of disguised shell scripts, executables, or HTML/JavaScript payloads.
2. **Directory Traversal**: Crafting storage keys (e.g. `../../etc/passwd`) to read or overwrite system files.
3. **Multi-Tenant Leakage**: An NGO viewing or altering evidence submitted by a different NGO.
4. **Data Destruction & Evidence Tampering**: Malicious actors deleting or secretly modifying inspection records.
5. **Unauthorized Download Leaks**: Public donors or competitors accessing confidential procurement quotes or sensitive field identities.

---

## 2. Ingestion Defense Layers

### A. Strict Binary Magic-Byte Inspection
CharityChain does not rely on file extensions or HTTP headers for file validation. Every file payload is inspected byte-by-byte:

```python
def _validate_magic_bytes(file_bytes: bytes, mime_type: str) -> bool:
    if not file_bytes:
        return False
    if mime_type == "application/pdf":
        return file_bytes.startswith(b"%PDF")
    elif mime_type == "image/jpeg":
        return file_bytes.startswith(b"\xff\xd8\xff")
    elif mime_type == "image/png":
        return file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    return False  # All unknown types are strictly rejected
```

Any mismatched or unknown file header raises `ValidationException` (HTTP 422) and is discarded before writing to storage.

### B. Filename Sanitization & Collision Prevention
Original filenames are stripped of path separators, control characters, and unsafe symbols:
```python
clean = os.path.basename(filename)
clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean)
unique_name = f"{file_hash[:16]}_{clean}"
```
This guarantees that uploaded files can never overwrite existing files or execute directory jumping.

### C. Resource Exhaustion Protections
- **Single File Limit**: Checked before reading full stream; enforced against `settings.MAX_UPLOAD_SIZE_MB` (default: 10.0 MB).
- **Batch Size Limit**: Requests exceeding `settings.MAX_UPLOAD_BATCH_SIZE` (default: 10 files) are rejected atomically.
- **Empty File Guard**: Files with zero bytes are immediately rejected.

---

## 3. Path Traversal & Containment Guarantees

In local storage mode, storage keys are canonicalized and verified against root boundaries:

```python
clean_key = os.path.normpath(storage_key).replace("\\", "/").lstrip("/")
target = (self.base_dir / clean_key).resolve()
try:
    target.relative_to(self.base_dir)
except ValueError:
    raise ValidationException("Path traversal attempt detected in storage key.")
```

Any path resolving outside `self.base_dir` immediately halts execution and records a security audit warning.

---

## 4. Multi-Tenant Access Control

Access control rules are strictly enforced at the domain service layer (`EvidenceService`):

| Action | Allowed Roles | Preconditions |
| :--- | :--- | :--- |
| **Upload Evidence** | `OWNER`, `ADMIN`, `MEMBER` (NGO) \| `PLATFORM_ADMIN` | Uploader must be an active member of the organization owning the project. Milestone must be active. |
| **Upload Version** | `OWNER`, `ADMIN` (NGO) \| `PLATFORM_ADMIN` | Parent evidence must belong to an active organization project. |
| **Verify Integrity** | `OWNER`, `ADMIN`, `MEMBER`, `REVIEWER`, `PLATFORM_ADMIN` | Live SHA-256 calculation available to authorized stakeholders. |
| **Download Raw Content** | `OWNER`, `ADMIN`, `MEMBER`, `REVIEWER`, `PLATFORM_ADMIN` | Confidential field files are restricted from unauthenticated or regular donor accounts. |
| **Soft-Archive** | `OWNER`, `ADMIN` (NGO) \| `PLATFORM_ADMIN` | Permitted only by organization admins or platform admins. |

---

## 5. Non-Destructive Archival Policy

To guarantee evidence provenance:
- **No Physical File Deletion**: The `archive_evidence` endpoint sets `is_archived = True` and records `archived_at`. The physical bytes on local disk or S3 bucket are **never deleted**.
- **Audit Persistence**: Archival is recorded in the SHA-256 audit ledger as an `ARCHIVED` event with the operator ID and justification.
- **Permanent Availability for Audits**: Even archived evidence can be retrieved by platform reviewers and admins for historical investigations.
