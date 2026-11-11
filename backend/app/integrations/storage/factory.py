"""
factory.py - Storage Provider Factory & Dependency Injector
===========================================================
Instantiates and configures the appropriate storage provider (Local, S3, MinIO)
based on application configuration and environment settings.
"""

from typing import Optional
from backend.app.core.config import settings
from backend.app.integrations.storage.base import StorageProviderInterface
from backend.app.integrations.storage.local import LocalStorageProvider
from backend.app.integrations.storage.s3 import S3StorageProvider

_storage_instance: Optional[StorageProviderInterface] = None


def get_storage_provider() -> StorageProviderInterface:
    """Returns configured storage provider singleton."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    provider_type = settings.STORAGE_PROVIDER.lower()
    if provider_type in ("s3", "minio"):
        _storage_instance = S3StorageProvider()
    else:
        _storage_instance = LocalStorageProvider(base_dir=settings.STORAGE_LOCAL_DIR)

    return _storage_instance


def set_storage_provider(provider: Optional[StorageProviderInterface]) -> None:
    """Overrides default storage provider for unit and integration testing."""
    global _storage_instance
    _storage_instance = provider


def reset_storage_provider() -> None:
    """Resets storage provider to None so factory re-evaluates configuration."""
    global _storage_instance
    _storage_instance = None
