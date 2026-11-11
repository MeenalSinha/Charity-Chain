"""
s3.py - AWS S3 & MinIO Storage Provider Implementation
======================================================
Secure cloud and object storage provider implementation supporting AWS S3
and S3-compatible endpoints (MinIO). Generates time-bounded presigned URLs,
enforces cryptographic integrity, and manages object lifecycles.
"""

import hashlib
from typing import Tuple, Optional, Dict, Any
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException, NotFoundException, InternalServerException
from backend.app.integrations.storage.base import StorageProviderInterface
from backend.app.integrations.storage.local import _sanitize_filename, _validate_magic_bytes


class S3StorageProvider(StorageProviderInterface):
    """AWS S3 and MinIO object storage provider."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        client: Optional[Any] = None
    ):
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.endpoint_url = endpoint_url or settings.S3_ENDPOINT_URL
        self.access_key = access_key or settings.S3_ACCESS_KEY
        self.secret_key = secret_key or settings.S3_SECRET_KEY
        self.region = region or settings.S3_REGION

        if client is not None:
            self._s3 = client
        else:
            client_kwargs: Dict[str, Any] = {
                "service_name": "s3",
                "region_name": self.region,
                "config": Config(signature_version="s3v4")
            }
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if self.access_key and self.secret_key:
                client_kwargs["aws_access_key_id"] = self.access_key
                client_kwargs["aws_secret_access_key"] = self.secret_key

            self._s3 = boto3.client(**client_kwargs)

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str, str, int]:
        """
        Validates and uploads file bytes to S3/MinIO bucket.
        Returns: (storage_key, presigned_url, sha256_hash, file_size_bytes)
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

        if subfolder:
            clean_sub = _sanitize_filename(subfolder)
            storage_key = f"{clean_sub}/{unique_name}"
        else:
            storage_key = unique_name

        try:
            self._s3.put_object(
                Bucket=self.bucket_name,
                Key=storage_key,
                Body=file_bytes,
                ContentType=mime_type,
                Metadata={
                    "sha256": file_hash,
                    "original_filename": clean_name
                }
            )
        except ClientError as e:
            raise InternalServerException(f"Failed to upload object to storage: {e}")

        presigned_url = self.generate_access_url(storage_key)
        return storage_key, presigned_url, file_hash, file_size

    def get_file(self, storage_key: str) -> bytes:
        try:
            response = self._s3.get_object(Bucket=self.bucket_name, Key=storage_key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise NotFoundException(f"Requested storage object '{storage_key}' not found.")
            raise InternalServerException(f"Storage retrieval error: {e}")

    def delete_file(self, storage_key: str) -> bool:
        try:
            self._s3.delete_object(Bucket=self.bucket_name, Key=storage_key)
            return True
        except ClientError as e:
            raise InternalServerException(f"Storage deletion error: {e}")

    def generate_access_url(self, storage_key: str, expires_in_seconds: int = 3600) -> str:
        expires_in = min(max(expires_in_seconds, 60), 86400)
        try:
            url = self._s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": storage_key
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            raise InternalServerException(f"Failed to generate presigned URL: {e}")

    def verify_content_integrity(self, storage_key: str, expected_sha256: str) -> bool:
        content = self.get_file(storage_key)
        computed = hashlib.sha256(content).hexdigest()
        return computed.lower() == expected_sha256.lower()

    def get_metadata(self, storage_key: str) -> Dict[str, Any]:
        try:
            res = self._s3.head_object(Bucket=self.bucket_name, Key=storage_key)
            return {
                "size": res.get("ContentLength", 0),
                "content_type": res.get("ContentType", ""),
                "etag": res.get("ETag", "").strip('"'),
                "sha256": res.get("Metadata", {}).get("sha256", ""),
                "storage_provider": "s3"
            }
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise NotFoundException(f"Storage object '{storage_key}' not found.")
            raise InternalServerException(f"Storage metadata error: {e}")

