"""
base.py - Abstract Storage Provider Interface
=============================================
Contract for object and file storage providers (Local disk, AWS S3, MinIO).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional


class StorageProviderInterface(ABC):
    """Abstract interface decoupling file persistence from specific cloud or local storage."""

    @abstractmethod
    async def upload_file(self, file_bytes: bytes, filename: str, mime_type: str) -> Tuple[str, str]:
        """
        Persists file bytes and returns (storage_key_or_cid, public_or_internal_url).
        """
        pass

    @abstractmethod
    async def get_file(self, storage_key: str) -> bytes:
        """Retrieves raw file bytes by key."""
        pass

    @abstractmethod
    async def delete_file(self, storage_key: str) -> bool:
        """Deletes file from storage."""
        pass

    @abstractmethod
    async def generate_access_url(self, storage_key: str, expires_in_seconds: int = 3600) -> str:
        """Generates a secure presigned access URL."""
        pass

    @abstractmethod
    async def verify_content_integrity(self, storage_key: str, expected_sha256: str) -> bool:
        """Verifies stored content matches cryptographic hash."""
        pass

