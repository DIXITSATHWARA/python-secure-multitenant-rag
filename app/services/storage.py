import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.core.config import Settings


@dataclass(frozen=True)
class StoredObject:
    key: str
    modified: datetime  # timezone-aware


class StorageBackend(Protocol):
    """Where original files live. Local disk in dev; S3-compatible storage in production."""

    async def save(self, key: str, data: bytes) -> None: ...

    async def read(self, key: str) -> bytes:
        """Raises FileNotFoundError when the key does not exist."""
        ...

    async def delete(self, key: str) -> None:
        """Must be idempotent: deleting a missing key is not an error."""
        ...

    async def list_keys(self) -> list[StoredObject]:
        """Every stored object. Used by the orphan sweep."""
        ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        # Defense in depth: keys are server-generated, but never trust that blindly.
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid storage key")
        return path

    async def save(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._save_sync, key, data)

    def _save_sync(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: readers never see a half-written file

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, True)  # missing_ok=True

    async def list_keys(self) -> list[StoredObject]:
        return await asyncio.to_thread(self._list_sync)

    def _list_sync(self) -> list[StoredObject]:
        found: list[StoredObject] = []
        for path in self.root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            except OSError:
                continue  # removed while we were listing
            found.append(StoredObject(path.relative_to(self.root).as_posix(), modified))
        return found


def _is_not_found(exc: Exception) -> bool:
    """Recognize S3's "no such key" without importing botocore."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    return str(response.get("Error", {}).get("Code")) in {"NoSuchKey", "404", "NotFound"}


class S3Storage:
    """S3-compatible object storage. boto3 is synchronous, so each call runs in a thread."""

    def __init__(self, client: Any, bucket: str, prefix: str = "") -> None:
        self._client = client
        self._bucket = bucket
        cleaned = prefix.strip("/")
        self._prefix = f"{cleaned}/" if cleaned else ""

    def _key(self, key: str) -> str:
        return self._prefix + key

    async def save(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(
            self._client.put_object, Bucket=self._bucket, Key=self._key(key), Body=data
        )

    async def read(self, key: str) -> bytes:
        try:
            response = await asyncio.to_thread(
                self._client.get_object, Bucket=self._bucket, Key=self._key(key)
            )
        except Exception as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(key) from None
            raise
        return await asyncio.to_thread(response["Body"].read)

    async def delete(self, key: str) -> None:
        # S3 answers 204 even when the key is missing, so this is naturally idempotent.
        await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=self._key(key))

    async def list_keys(self) -> list[StoredObject]:
        return await asyncio.to_thread(self._list_sync)

    def _list_sync(self) -> list[StoredObject]:
        found: list[StoredObject] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self._prefix and not key.startswith(self._prefix):
                    continue
                found.append(StoredObject(key[len(self._prefix) :], obj["LastModified"]))
        return found


def _build_s3_client(settings: Settings) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError(
            'STORAGE_BACKEND=s3 needs boto3. Install it with: pip install ".[s3]"'
        ) from None

    options: dict[str, Any] = {}
    if settings.s3_region:
        options["region_name"] = settings.s3_region
    if settings.s3_endpoint_url:
        options["endpoint_url"] = settings.s3_endpoint_url
    secret = settings.s3_secret_access_key.get_secret_value()
    if settings.s3_access_key_id and secret:
        options["aws_access_key_id"] = settings.s3_access_key_id
        options["aws_secret_access_key"] = secret
    config = Config(
        retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=5, read_timeout=30
    )
    return boto3.client("s3", config=config, **options)


_s3_cache: dict[tuple[str, ...], StorageBackend] = {}


def build_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(Path(settings.local_storage_path))
    if settings.storage_backend == "s3":
        # Built once per configuration: creating a boto3 client on every request is slow.
        cache_key = (
            settings.s3_bucket,
            settings.s3_prefix,
            settings.s3_region,
            settings.s3_endpoint_url,
        )
        if cache_key not in _s3_cache:
            client = _build_s3_client(settings)
            _s3_cache[cache_key] = S3Storage(client, settings.s3_bucket, settings.s3_prefix)
        return _s3_cache[cache_key]
    raise RuntimeError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")
