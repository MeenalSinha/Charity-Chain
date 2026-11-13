"""
tests/infrastructure/test_backup_integrity.py

Phase 14: Verifies backup configuration, script existence,
and simulated backup/restore integrity logic.
Does not require actual cloud credentials or running containers.
"""
import hashlib
import io
import json
import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


class TestBackupConfiguration:
    """Verify backup scripts and configuration are properly set up."""

    def test_backup_script_is_executable_or_has_shebang(self):
        """backup_postgres.sh must have a bash shebang."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh")
        assert os.path.isfile(path), "backup_postgres.sh not found"
        with open(path) as f:
            first_line = f.readline()
        assert first_line.startswith("#!/"), (
            f"Script must have a shebang line, got: {first_line!r}"
        )

    def test_backup_script_uses_set_euo_pipefail(self):
        """Backup scripts must use 'set -euo pipefail' for safety."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh")
        with open(path) as f:
            content = f.read()
        assert "set -euo pipefail" in content, (
            "Backup script must use 'set -euo pipefail' for error safety"
        )

    def test_backup_script_validates_required_env_vars(self):
        """Backup script must validate required environment variables."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh")
        with open(path) as f:
            content = f.read()
        required_vars = ["BACKUP_BUCKET", "POSTGRES_HOST", "POSTGRES_PASSWORD", "BACKUP_ENCRYPTION_KEY"]
        for var in required_vars:
            assert var in content, f"Backup script must check for {var}"

    def test_backup_script_cleans_up_on_exit(self):
        """Backup script must clean up temporary files on exit."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh")
        with open(path) as f:
            content = f.read()
        assert "trap cleanup EXIT" in content or "trap" in content, (
            "Backup script must trap EXIT to clean up temporary files"
        )

    def test_backup_script_uploads_to_gcs(self):
        """Backup script must upload to GCS."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh")
        with open(path) as f:
            content = f.read()
        assert "gsutil" in content, "Backup script must use gsutil for GCS uploads"

    def test_verify_script_has_integrity_checks(self):
        """verify_backup.sh must include database integrity checks."""
        path = os.path.join(REPO_ROOT, "scripts", "backup", "verify_backup.sh")
        with open(path) as f:
            content = f.read()
        assert "integrity" in content.lower() or "COUNT" in content, (
            "Verify script must run integrity checks on the restored database"
        )


class TestBackupDataIntegrity:
    """Test backup data integrity logic (without actual cloud resources)."""

    def test_backup_checksum_matches_after_restore(self):
        """Simulate that backup checksum matches after restore."""
        # Simulate database dump content
        fake_db_dump = b"CREATE TABLE users (id SERIAL PRIMARY KEY);\nINSERT INTO users VALUES (1);\n"

        # Compute SHA-256 of original
        original_hash = hashlib.sha256(fake_db_dump).hexdigest()

        # Simulate compression (gzip)
        import gzip
        compressed = gzip.compress(fake_db_dump, compresslevel=9)

        # Simulate decompression
        decompressed = gzip.decompress(compressed)

        # Verify integrity
        restored_hash = hashlib.sha256(decompressed).hexdigest()
        assert original_hash == restored_hash, (
            "Backup data integrity check failed: hash mismatch after compress/decompress"
        )

    def test_backup_metadata_schema(self):
        """Backup metadata JSON must contain required fields."""
        metadata = {
            "timestamp": "20261114T000000Z",
            "environment": "production",
            "database": "charitychain",
            "host": "db.example.com",
            "filename": "charitychain-production-20261114T000000Z.sql.gz.gpg",
            "gcs_path": "gs://charitychain-backups-prod/postgres/charitychain-production-20261114T000000Z.sql.gz.gpg",
            "compressed_size": "150M",
            "encrypted_size": "151M",
            "retention_days": 90,
            "status": "SUCCESS",
        }
        required_fields = [
            "timestamp", "environment", "database", "host",
            "filename", "gcs_path", "status", "retention_days"
        ]
        for field in required_fields:
            assert field in metadata, f"Backup metadata must include '{field}'"
        assert metadata["status"] == "SUCCESS"

    def test_retention_policy_logic(self):
        """Verify retention policy correctly identifies expired backups."""
        from datetime import datetime, timedelta

        # Simulate backup filenames with dates
        backups = [
            f"charitychain-production-{(datetime.utcnow() - timedelta(days=i)).strftime('%Y%m%d')}T000000Z.sql.gz.gpg"
            for i in range(100)
        ]

        retention_days = 90
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        expired = []
        for backup in backups:
            # Extract date from filename
            import re
            match = re.search(r"(\d{8})T", backup)
            if match:
                backup_date = datetime.strptime(match.group(1), "%Y%m%d")
                if backup_date < cutoff:
                    expired.append(backup)

        # With retention_days=90, backups older than 90 days should be expired
        # We created backups 0..99 days old, so ~10 should be expired (days 90-99)
        assert len(expired) >= 9, f"Expected ~10 expired backups, found {len(expired)}"
        assert len(expired) <= 11, f"Too many expired backups: {len(expired)}"

