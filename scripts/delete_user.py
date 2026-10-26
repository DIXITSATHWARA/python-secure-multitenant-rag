"""Delete an account and all of its data, in two steps.

Run:  python -m scripts.delete_user EMAIL              (step 1: lock account, queue cleanup)
      python -m scripts.delete_user EMAIL --finalize   (step 2: after the worker finished)
"""

import argparse
import asyncio
import sys

from app.db.session import AsyncSessionLocal, engine
from app.workers.admin_ops import finalize_user_deletion, request_user_deletion


async def main(email: str, finalize: bool) -> int:
    try:
        if not finalize:
            count = await request_user_deletion(AsyncSessionLocal, email)
            print(
                f"Account locked, {count} document(s) queued for deletion.\n"
                "Wait for the worker (python -m scripts.ops_report shows the queue), "
                "then re-run with --finalize."
            )
            return 0
        if await finalize_user_deletion(AsyncSessionLocal, email):
            print("User deleted.")
            return 0
        print("Documents still exist: the worker has not finished. Try again shortly.")
        return 1
    except (LookupError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("email")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.email, args.finalize)))
