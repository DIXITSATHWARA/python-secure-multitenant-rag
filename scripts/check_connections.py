"""Verify PostgreSQL, Qdrant Cloud and OpenAI connectivity.

Run from the project root:  python -m scripts.check_connections
"""

import asyncio

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


async def check_postgres() -> None:
    s = get_settings()
    engine = create_async_engine(s.database_url)
    try:
        async with engine.connect() as conn:
            version = (await conn.execute(text("select version()"))).scalar_one()
        print(f"[OK] PostgreSQL: {version[:60]}")
    finally:
        await engine.dispose()


async def check_qdrant() -> None:
    s = get_settings()
    client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value())
    try:
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        print(f"[OK] Qdrant Cloud: {len(names)} collection(s) {names}")
    finally:
        await client.close()


async def check_openai() -> None:
    s = get_settings()
    client = AsyncOpenAI(
        api_key=s.openai_api_key.get_secret_value(), timeout=s.openai_timeout_seconds
    )
    resp = await client.embeddings.create(
        model=s.openai_embedding_model,
        input="connectivity test",
        dimensions=s.openai_embedding_dimensions,
    )
    dim = len(resp.data[0].embedding)
    ok = "OK" if dim == s.openai_embedding_dimensions else "MISMATCH"
    print(f"[{ok}] OpenAI embeddings: model={s.openai_embedding_model}, dimensions={dim}")


async def main() -> None:
    for name, check in [
        ("PostgreSQL", check_postgres),
        ("Qdrant", check_qdrant),
        ("OpenAI", check_openai),
    ]:
        try:
            await check()
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
