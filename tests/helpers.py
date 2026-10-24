from pathlib import Path

from app.services.storage import LocalStorage

SOME_ID = "00000000-0000-4000-8000-000000000001"


def files_on_disk(storage: LocalStorage) -> list[Path]:
    return [p for p in storage.root.rglob("*") if p.is_file()]
