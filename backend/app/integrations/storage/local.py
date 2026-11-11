"""
local.py - Local Disk Storage Provider Implementation
======================================================
Secure local filesystem implementation of StorageProviderInterface.
Enforces filename sanitization, magic bytes verification, file size limits,
and path traversal defenses with canonical path containment.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException, NotFoundException
from backend.app.integrations.storage.base import StorageProviderInterface


def _sanitize_filename(filename: str) -> str:
    """Removes path separators, whitespace, and dangerous characters."""
    clean = os.path.basename(filename)
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean)
    return clean or "document"


def _validate_magic_bytes(file_bytes: bytes, mime_type: str) -> bool:
    """Validates real file magic bytes matches declared MIME type."""
    if not file_bytes:
        return False
    if mime_type == "application/pdf":
        return file_bytes.startswith(b"%PDF")
    elif mime_type == "image/jpeg":
        return file_bytes.startswith(b"\xff\xd8\xff")
    elif mime_type == "image/png":
        return file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    return False


class LocalStorageProvider(StorageProviderInterface):
    """Local filesystem storage adapter adhering to StorageProviderInterface."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir or settings.STORAGE_LOCAL_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, storage_key: str) -> Path:
        """Resolves target file path and strictly blocks path traversal attempts."""
        clean_key = os.path.normpath(storage_key).replace("\\", "/").lstrip("/")
        target = (self.base_dir / clean_key).resolve()
        try:
            target.relative_to(self.base_dir)
        except ValueError:
            raise ValidationException("Path traversal attempt detected in storage key.")
        return target

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str, str, int]:
        """
        Validates, hashes, and stores file bytes to disk.
        Returns: (storage_key, relative_url, sha256_hash, file_size_bytes)
        """
        if not file_bytes:
            raise ValidationException("Uploaded file content cannot be empty.")

        max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
        file_size = len(file_bytes)
        if file_size > max_bytes:
            raise ValidationException(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed {settings.MAX_UPLOAD_SIZE_MB} MB."
            )

        if mime_type not in settings.ALLOWED_MIME_TYPES:
            raise ValidationException(
                f"Unsupported MIME type '{mime_type}'. Allowed: {settings.ALLOWED_MIME_TYPES}"
            )

        if not _validate_magic_bytes(file_bytes, mime_type):
            raise ValidationException("File contents do not match the declared MIME type magic signature.")

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        clean_name = _sanitize_filename(filename)
        unique_name = f"{file_hash[:16]}_{clean_name}"

        dest_dir = self.base_dir
        if subfolder:
            clean_sub = _sanitize_filename(subfolder)
            dest_dir = self.base_dir / clean_sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            storage_key = f"{clean_sub}/{unique_name}"
        else:
            storage_key = unique_name

        dest_path = self._resolve_safe_path(storage_key)
        dest_path.write_bytes(file_bytes)

        relative_url = f"/api/v1/evidence/{storage_key}/download"
        return storage_key, relative_url, file_hash, file_size

    def save_file_sync(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str, str, int]:
        """Alias for upload_file to preserve backward compatibility."""
        return self.upload_file(file_bytes, filename, mime_type, subfolder)

    def get_file(self, storage_key: str) -> bytes:
        target = self._resolve_safe_path(storage_key)
        if not target.exists() or not target.is_file():
            raise NotFoundException("Requested document file not found on disk.")
        return target.read_bytes()

    def get_file_sync(self, storage_key: str) -> bytes:
        return self.get_file(storage_key)

    def delete_file(self, storage_key: str) -> bool:
        target = self._resolve_safe_path(storage_key)
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False

    def delete_file_sync(self, storage_key: str) -> bool:
        return self.delete_file(storage_key)

    def generate_access_url(self, storage_key: str, expires_in_seconds: int = 3600) -> str:
        # Local storage serves internally via authorized endpoint
        return f"/api/v1/evidence/{storage_key}/download"

    def verify_content_integrity(self, storage_key: str, expected_sha256: str) -> bool:
        data = self.get_file(storage_key)
        computed = hashlib.sha256(data).hexdigest()
        return computed.lower() == expected_sha256.lower()

    def get_metadata(self, storage_key: str) -> Dict[str, Any]:
        target = self._resolve_safe_path(storage_key)
        if not target.exists() or not target.is_file():
            raise NotFoundException("Requested file not found.")
        stat = target.stat()
        return {
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "storage_provider": "local"
        }


# Singleton instance for NGO verification documents
document_storage = LocalStorageProvider(base_dir=settings.STORAGE_DOCUMENTS_DIR)
