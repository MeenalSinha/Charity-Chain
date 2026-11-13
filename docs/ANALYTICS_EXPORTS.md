# CharityChain — Analytics Data Exports & Formula Injection Hardening

## 1. Overview & Export Formats
CharityChain supports automated and on-demand exports of analytics datasets and summary reports. Supported export formats include:
- **CSV Data Tables**: Structured tabular exports for donations, projects, and milestone progress.
- **Plain Text / Markdown Reports**: Human-readable executive performance summaries and annual donor tax statements.
- **Asynchronous Snapshot Models**: Persistent `analytics_snapshots` and `analytics_export_jobs` records tracking export requests.

---

## 2. CSV Formula Injection (CWE-1236) Defense
Spreadsheet applications (such as Microsoft Excel, LibreOffice Calc, Apple Numbers, and Google Sheets) automatically interpret and execute cells starting with formula trigger symbols:
- `=` (Formula evaluation)
- `+` (Arithmetic addition / formula trigger)
- `-` (Arithmetic negation / formula trigger)
- `@` (Function call in Excel / LibreOffice)
- `\t` (Tab character used to evade filters)
- `\r` (Carriage return character used to evade filters)

If user-supplied strings (such as donor names, notes, project descriptions, or external references) contain malicious formulas, opening the exported CSV can execute arbitrary operating system commands via DDE or leak sensitive environment tokens.

### Sanitization Implementation
The `AnalyticsExportService.sanitize_for_csv` function inspects all cell values prior to writing:
```python
@staticmethod
def sanitize_for_csv(val: Any) -> Any:
    """Neutralizes formula injection attacks in spreadsheet exports."""
    if val is None:
        return ""
    s = str(val)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{s}"
    return s
```
By prepending a single quote (`'`), spreadsheet engines treat the cell strictly as text, neutralizing execution while preserving readability.

---

## 3. Asynchronous Export Jobs
Long-running export operations are tracked via the `analytics_export_jobs` table:
- **Status Lifecycle**: `PENDING -> PROCESSING -> COMPLETED | FAILED`
- **Audit Columns**: `requested_by_id`, `created_at`, `completed_at`, `file_format`, `file_storage_key`, `checksum_sha256`.
- **Retention**: Exported files are retained in secure storage (S3/MinIO) with time-limited signed download URLs.
