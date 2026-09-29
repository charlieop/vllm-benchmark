"""Optional S3 archival.  Local run data is always left untouched."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveResult:
    uploaded: bool
    bucket: str
    key: str
    error: str | None = None


class S3Archive:
    """Upload a compressed copy of one completed local run directory.

    Supplying a boto3-compatible ``client`` makes this usable without importing
    boto3 in tests.  If boto3 is absent, :meth:`archive_run` returns a failed
    result instead of affecting generation or local evaluation output.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must be non-empty")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self._client = client

    def _key_for(self, run_dir: Path) -> str:
        name = f"{run_dir.parent.name}/{run_dir.name}.tar.gz"
        return f"{self.prefix}/{name}" if self.prefix else name

    def _client_or_error(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 archival; install it or provide an S3 client") from exc
        self._client = boto3.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url)
        return self._client

    def archive_run(self, local_run_dir: str | Path) -> ArchiveResult:
        run_dir = Path(local_run_dir)
        key = self._key_for(run_dir)
        if not run_dir.is_dir():
            return ArchiveResult(False, self.bucket, key, f"local run directory does not exist: {run_dir}")
        try:
            import tarfile
            with tempfile.TemporaryDirectory(prefix="visionbench-s3-") as temporary:
                archive_path = Path(temporary) / f"{run_dir.name}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    for item in sorted(run_dir.rglob("*")):
                        if item.is_file():
                            archive.add(item, arcname=str(item.relative_to(run_dir)))
                self._client_or_error().upload_file(archive_path, self.bucket, key)
            return ArchiveResult(True, self.bucket, key)
        except Exception as exc:
            return ArchiveResult(False, self.bucket, key, str(exc))
