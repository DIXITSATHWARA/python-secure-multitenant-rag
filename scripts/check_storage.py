"""Write, read, list and delete a test object.  Run: python -m scripts.check_storage"""

import asyncio
import uuid

from app.core.config import get_settings
from app.services.storage import build_storage


async def main() -> None:
    settings = get_settings()
    storage = build_storage(settings)
    key = f"healthcheck/{uuid.uuid4()}.txt"  # not a managed key shape, so the sweep ignores it
    await storage.save(key, b"hello")
    assert await storage.read(key) == b"hello"
    assert key in {obj.key for obj in await storage.list_keys()}
    await storage.delete(key)
    try:
        await storage.read(key)
    except FileNotFoundError:
        print(f"[OK] storage backend '{settings.storage_backend}': write, read, list, delete")
    else:
        raise SystemExit("[FAIL] the object still exists after delete")


if __name__ == "__main__":
    asyncio.run(main())
