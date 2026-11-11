"""
test_storage_providers.py - Storage Provider Adapters Unit Tests
================================================================
Verifies LocalStorageProvider, S3StorageProvider, path-traversal prevention,
and factory configuration.
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from backend.app.core.exceptions import ValidationException, NotFoundException
from backend.app.integrations.storage.local import LocalStorageProvider
from backend.app.integrations.storage.s3 import S3StorageProvider
from backend.app.integrations.storage.factory import (
    get_storage_provider, set_storage_provider, reset_storage_provider
)


def test_local_storage_lifecycle(tmp_path):
    """Verify local disk storage upload, get, verify, and delete operations."""
    provider = LocalStorageProvider(base_dir=tmp_path)
    content = b"%PDF-1.4 sample evidence document bytes"
    filename = "report.pdf"
    mime_type = "application/pdf"

    # Upload
    key, url, sha256_hash, size = provider.upload_file(content, filename, mime_type, subfolder="m1")
    assert "m1/" in key
    assert size == len(content)
    assert len(sha256_hash) == 64

    # Get
    retrieved = provider.get_file(key)
    assert retrieved == content

    # Verify integrity
    assert provider.verify_content_integrity(key, sha256_hash) is True
    assert provider.verify_content_integrity(key, "0" * 64) is False

    # Metadata
    meta = provider.get_metadata(key)
    assert meta["size"] == len(content)

    # Delete
    assert provider.delete_file(key) is True
    with pytest.raises(NotFoundException):
        provider.get_file(key)


def test_local_storage_blocks_path_traversal(tmp_path):
    """Verify LocalStorageProvider strictly blocks path traversal in keys."""
    provider = LocalStorageProvider(base_dir=tmp_path)
    
    with pytest.raises(ValidationException, match="Path traversal"):
        provider._resolve_safe_path("../../../windows/system32/cmd.exe")


def test_local_storage_rejects_empty_and_oversized_files(tmp_path):
    """Verify upload rejects empty files and oversize payloads."""
    provider = LocalStorageProvider(base_dir=tmp_path)
    
    with pytest.raises(ValidationException, match="cannot be empty"):
        provider.upload_file(b"", "empty.png", "image/png")

    # 11MB file with default limit 10MB
    oversized = b"\x89PNG\r\n\x1a\n" + (b"0" * (11 * 1024 * 1024))
    with pytest.raises(ValidationException, match="exceeds maximum"):
        provider.upload_file(oversized, "huge.png", "image/png")


def test_s3_storage_with_mock_client():
    """Verify S3StorageProvider interactions with boto3 client."""
    mock_client = MagicMock()
    mock_client.put_object.return_value = {"ETag": '"test-etag"'}
    mock_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/test-bucket/evidence.jpg?presigned=1"
    
    mock_body = MagicMock()
    content = b"\xff\xd8\xff\xe0JFIF photo payload"
    mock_body.read.return_value = content
    mock_client.get_object.return_value = {"Body": mock_body}
    mock_client.head_object.return_value = {
        "ContentLength": len(content),
        "ContentType": "image/jpeg",
        "ETag": '"test-etag"',
        "Metadata": {"sha256": "testhash"}
    }

    provider = S3StorageProvider(
        bucket_name="test-bucket",
        client=mock_client
    )

    # Upload
    key, url, h, sz = provider.upload_file(content, "photo.jpg", "image/jpeg")
    assert mock_client.put_object.called
    assert "https://s3.amazonaws.com" in url
    assert sz == len(content)

    # Get
    data = provider.get_file(key)
    assert data == content

    # Verify integrity
    assert provider.verify_content_integrity(key, h) is True

    # Delete
    assert provider.delete_file(key) is True
    assert mock_client.delete_object.called


def test_storage_factory_override():
    """Verify get_storage_provider and set_storage_provider override mechanism."""
    reset_storage_provider()
    default_p = get_storage_provider()
    assert default_p is not None

    custom_mock = MagicMock(spec=LocalStorageProvider)
    set_storage_provider(custom_mock)
    assert get_storage_provider() is custom_mock

    reset_storage_provider()
    assert get_storage_provider() is not custom_mock
