import asyncio
import os
from pathlib import Path
from typing import Protocol

from app.core.config import Settings


class StorageBackend(Protocol):
    """Where original files live. Local disk in dev; an S3-compatible class in production."""

    async def save(self, key: str, data: bytes) -> None: ...

    async def read(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None:
        """Must be idempotent: deleting a missing key is not an error."""
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


def build_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(Path(settings.local_storage_path))
    raise RuntimeError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")
