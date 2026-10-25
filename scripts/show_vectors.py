"""Inspect the Qdrant collection.  Run: python -m scripts.show_vectors [document_id]"""

import asyncio
import sys

from qdrant_client import models

from app.core.config import get_settings
from app.services.clients import build_qdrant_client


async def main(document_id: str | None) -> None:
    settings = get_settings()
    client = build_qdrant_client(settings)
    name = settings.qdrant_collection
    try:
        if not await client.collection_exists(name):
            print(f"Collection '{name}' does not exist yet (the worker or API creates it).")
            return
        info = await client.get_collection(name)
        vectors = info.config.params.vectors
        print(f"collection: {name}")
        print(f"points:     {info.points_count}")
        print(f"vectors:    size={vectors.size} distance={vectors.distance}")
        print("payload indexes:")
        for field, schema in (info.payload_schema or {}).items():
            print(f"  {field}: {schema.data_type}")

        flt = None
        if document_id:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        points, _ = await client.scroll(
            collection_name=name, scroll_filter=flt, limit=5, with_payload=True, with_vectors=False
        )
        print(f"\nfirst {len(points)} point(s){' for ' + document_id if document_id else ''}:")
        for p in points:
            payload = p.payload or {}
            preview = str(payload.get("text", ""))[:70].replace("\n", " ")
            print(f"  chunk {payload.get('chunk_index')}: {preview}...")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
