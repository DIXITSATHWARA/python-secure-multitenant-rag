"""Measure retrieval quality for one user.

Run:  python -m scripts.eval_retrieval you@gmail.com eval_cases.json

eval_cases.json is a list of cases. "expect" is text that the correct chunk contains.
Use null for questions your documents should NOT be able to answer:
  [{"question": "How long do cats sleep?", "expect": "12 and 16 hours"},
   {"question": "What is the capital of France?", "expect": null}]
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal, engine
from app.models import User
from app.services.clients import build_openai_client, build_qdrant_client
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


async def main(email: str, cases_path: str) -> None:
    settings = get_settings()
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    async with AsyncSessionLocal() as db:
        user_id = (
            await db.execute(select(User.id).where(User.email == email.lower()))
        ).scalar_one_or_none()
    if user_id is None:
        print(f"No user with email {email}")
        await engine.dispose()
        return

    openai_client = build_openai_client(settings)
    qdrant = build_qdrant_client(settings)
    vectors = VectorService(qdrant, settings)
    embeddings = EmbeddingService(openai_client, settings)
    k = settings.retrieval_top_k
    positives = hits = 0
    reciprocal_sum = 0.0
    try:
        for case in cases:
            vector = await embeddings.embed_query(case["question"])
            # threshold 0 so we see the raw scores, which is what you tune the threshold from
            found = await vectors.search(
                user_id=user_id, vector=vector, limit=k, score_threshold=0.0
            )
            top = found[0].score if found else 0.0
            expect = case.get("expect")
            if expect is None:
                print(f"NEGATIVE  top score {top:.2f}  {case['question']}")
                continue
            positives += 1
            rank = next(
                (i for i, c in enumerate(found, 1) if expect.lower() in c.text.lower()), None
            )
            if rank:
                hits += 1
                reciprocal_sum += 1 / rank
            label = f"HIT@{rank}" if rank else "MISS"
            print(f"{label:<9} top score {top:.2f}  {case['question']}")
    finally:
        await qdrant.close()
        await openai_client.close()
        await engine.dispose()

    if positives:
        print(f"\nhit rate @{k}: {hits / positives:.0%}   MRR: {reciprocal_sum / positives:.2f}")
    print(f"current threshold: {settings.retrieval_score_threshold}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m scripts.eval_retrieval EMAIL CASES.json")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
