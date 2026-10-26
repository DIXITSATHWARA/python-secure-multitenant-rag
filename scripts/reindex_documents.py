"""Re-queue every completed document for processing (re-chunk and re-embed).

Run:  python -m scripts.reindex_documents           (dry run: only counts)
      python -m scripts.reindex_documents --apply

Documents are not searchable again until the worker has finished them.
"""

import argparse
import asyncio
import sys

from app.db.session import AsyncSessionLocal, engine
from app.workers.admin_ops import request_reindex


async def main(apply: bool) -> None:
    try:
        count = await request_reindex(AsyncSessionLocal, dry_run=not apply)
    finally:
        await engine.dispose()
    if apply:
        print(f"Queued {count} document(s) for re-processing. Watch: python -m scripts.ops_report")
    else:
        print(f"{count} completed document(s) would be re-processed. Re-run with --apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    asyncio.run(main(parser.parse_args().apply))
    sys.exit(0)
