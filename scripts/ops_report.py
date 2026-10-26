"""Queue and cleanup health check. Exit code 0 = healthy, 1 = needs attention.

Run:  python -m scripts.ops_report [--max-queue-age 600]
Schedule it every few minutes and alert on a non-zero exit code.
"""

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal, engine
from app.workers.ops import collect_ops_stats


async def main(max_queue_age: int) -> int:
    settings = get_settings()
    try:
        stats = await collect_ops_stats(AsyncSessionLocal, lease_seconds=settings.job_lease_seconds)
    finally:
        await engine.dispose()

    age = "none" if stats.oldest_due_age_seconds is None else f"{stats.oldest_due_age_seconds:.0f}s"
    print(
        f"jobs: queued={stats.queued} running={stats.running} failed={stats.failed}  "
        f"oldest runnable wait={age}  stale leases={stats.stale_running}  "
        f"stuck deletes={stats.stuck_deleting}"
    )
    problems = stats.problems(max_queue_age_seconds=max_queue_age)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-queue-age", type=int, default=600)
    sys.exit(asyncio.run(main(parser.parse_args().max_queue_age)))
