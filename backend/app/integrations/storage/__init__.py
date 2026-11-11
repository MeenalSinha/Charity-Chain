"""
Storage module package exports.
"""

from backend.app.integrations.storage.base import StorageProviderInterface
from backend.app.integrations.storage.local import (
    LocalStorageProvider,
    document_storage,
    _sanitize_filename,
    _validate_magic_bytes
)
from backend.app.integrations.storage.s3 import S3StorageProvider
from backend.app.integrations.storage.factory import (
    get_storage_provider,
    set_storage_provider,
    reset_storage_provider
)

__all__ = [
    "StorageProviderInterface",
    "LocalStorageProvider",
    "S3StorageProvider",
    "document_storage",
    "_sanitize_filename",
    "_validate_magic_bytes",
    "get_storage_provider",
    "set_storage_provider",
    "reset_storage_provider"
]
