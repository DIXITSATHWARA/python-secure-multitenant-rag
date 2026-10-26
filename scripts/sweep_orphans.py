"""Find orphaned files and vectors. A dry run unless --apply is given.

Run:  python -m scripts.sweep_orphans [--apply] [--force] [--grace-hours 24]
"""

import argparse
import asyncio
import sys
from datetime import timedelta

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal, engine
from app.services.clients import build_qdrant_client
from app.services.storage import build_storage
from app.services.vector_service import VectorService
from app.workers.sweep import SweepRefused, sweep_orphans


def show(title: str, items: list[str]) -> None:
    print(f"{title}: {len(items)}")
    for item in items[:20]:
        print(f"  {item}")
    if len(items) > 20:
        print(f"  ... and {len(items) - 20} more")


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    qdrant = build_qdrant_client(settings)
    try:
        report = await sweep_orphans(
            AsyncSessionLocal,
            build_storage(settings),
            VectorService(qdrant, settings),
            apply=args.apply,
            force=args.force,
            grace=timedelta(hours=args.grace_hours),
        )
    except SweepRefused as exc:
        print(f"REFUSED: {exc}")
        return 2
    finally:
        await qdrant.close()
        await engine.dispose()

    show("orphan files", report.orphan_files)
    show(
        "orphan vector documents (user, document)",
        [f"{user} {doc}" for user, doc in report.orphan_vector_documents],
    )
    if args.apply:
        print(
            f"deleted {report.files_deleted} file(s) and the vectors of "
            f"{report.vectors_deleted} document(s); {report.errors} error(s)"
        )
    else:
        print("\nDry run: nothing was deleted. Re-run with --apply to remove the items above.")
    return 1 if report.errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually delete the orphans")
    parser.add_argument("--force", action="store_true", help="skip the empty-database safety check")
    parser.add_argument(
        "--grace-hours", type=float, default=24, help="protect files newer than this"
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
