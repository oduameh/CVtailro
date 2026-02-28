"""
CVtailro Cloudflare R2 File Storage

Provides upload, presigned URL generation, and deletion for job artifacts
stored in Cloudflare R2 (S3-compatible). Falls back gracefully when R2
credentials are not configured.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from io import BytesIO
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2Storage:
    """Cloudflare R2 storage client with lazy initialization."""

    def __init__(self):
        self._client = None
        self._bucket = None

    def init_app(self, app):
        """Initialize R2 client from environment variables."""
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self._bucket = os.environ.get("R2_BUCKET_NAME", "cvtailro")

        if not all([account_id, access_key, secret_key]):
            logger.warning("R2 not configured — using local file storage")
            return

        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3},
            ),
            region_name="auto",
        )
        logger.info(f"R2 storage initialized: bucket={self._bucket}")

    @property
    def is_configured(self):
        """Return True if R2 credentials are set and client is ready."""
        return self._client is not None

    def upload_file(self, job_id, filename, file_path=None, file_data=None):
        """Upload a file to R2 under jobs/{job_id}/{filename}.

        Provide either file_path (path on disk) or file_data (bytes).
        Returns the R2 key.
        """
        if not self.is_configured:
            raise RuntimeError("R2 not configured")

        r2_key = f"jobs/{job_id}/{Path(filename).name}"
        ct = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        if file_path:
            self._client.upload_file(
                str(file_path),
                self._bucket,
                r2_key,
                ExtraArgs={"ContentType": ct},
            )
        elif file_data:
            self._client.upload_fileobj(
                BytesIO(file_data),
                self._bucket,
                r2_key,
                ExtraArgs={"ContentType": ct},
            )
        else:
            raise ValueError("file_path or file_data required")

        logger.info(f"Uploaded to R2: {r2_key}")
        return r2_key

    def generate_presigned_url(self, r2_key, expires_in=3600):
        """Generate a presigned URL for downloading a file from R2."""
        if not self.is_configured:
            raise RuntimeError("R2 not configured")
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": r2_key},
            ExpiresIn=expires_in,
        )

    def delete_job_files(self, job_id):
        """Delete all files under jobs/{job_id}/ in R2. Returns count deleted."""
        if not self.is_configured:
            return 0

        prefix = f"jobs/{job_id}/"
        deleted = 0
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                if objects:
                    self._client.delete_objects(
                        Bucket=self._bucket,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                    )
                    deleted += len(objects)
        except ClientError as e:
            logger.error(f"R2 delete failed: {e}")
        return deleted


r2_storage = R2Storage()
