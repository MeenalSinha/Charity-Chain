"""
base.py - Abstract Storage Provider Interface
=============================================
Contract for object and file storage providers (Local disk, AWS S3, MinIO).
Standardizes secure file upload, retrieval, deletion, presigned URL generation,
and cryptographic integrity validation across local and cloud environments.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class StorageProviderInterface(ABC):
    """Abstract interface decoupling file persistence from specific cloud or local storage."""

    @abstractmethod
    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str, str, int]:
        """
        Persists file bytes safely and returns:
        (storage_key, access_url_or_relative_path, sha256_hash, file_size_bytes)
        """
        pass

    @abstractmethod
    def get_file(self, storage_key: str) -> bytes:
        """Retrieves raw file bytes by storage key."""
        pass

    @abstractmethod
    def delete_file(self, storage_key: str) -> bool:
        """Deletes file from storage."""
        pass

    @abstractmethod
    def generate_access_url(self, storage_key: str, expires_in_seconds: int = 3600) -> str:
        """Generates a secure presigned or relative access URL."""
        pass

    @abstractmethod
    def verify_content_integrity(self, storage_key: str, expected_sha256: str) -> bool:
        """Verifies stored content matches cryptographic hash."""
        pass

    @abstractmethod
    def get_metadata(self, storage_key: str) -> Dict[str, Any]:
        """Retrieves metadata for stored object (e.g., size, content_type)."""
        pass
