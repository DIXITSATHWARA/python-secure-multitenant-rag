"""Worker entrypoint.  Run:  python -m app.workers.runner   (from the project root)"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time
from types import FrameType

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import configure_logging, request_id_ctx
from app.db.session import AsyncSessionLocal, engine
from app.services.clients import build_openai_client, build_qdrant_client
from app.services.embedding_service import EmbeddingService
from app.services.storage import build_storage
from app.services.vector_service import VectorService
from app.workers.maintenance import purge_old_rows, recover_stale_jobs
from app.workers.processor import DocumentProcessor, SessionFactory, claim_job

logger = logging.getLogger("app.worker")

MAINTENANCE_INTERVAL_SECONDS = 30
PURGE_INTERVAL_SECONDS = 3600


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    """Sleep, but wake immediately when shutdown is requested."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


class Worker:
    def __init__(
        self, settings: Settings, session_factory: SessionFactory, processor: DocumentProcessor
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._processor = processor

    async def run(self, stop: asyncio.Event) -> None:
        async with asyncio.TaskGroup() as group:
            for n in range(self._settings.worker_concurrency):
                group.create_task(self._job_loop(n, stop))
            group.create_task(self._maintenance_loop(stop))

    async def _job_loop(self, n: int, stop: asyncio.Event) -> None:
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{n}"
        while not stop.is_set():
            try:
                job = await claim_job(self._session_factory, worker_id)
            except Exception:
                logger.exception("Could not claim a job")
                await _sleep(stop, 5)
                continue
            if job is None:
                await _sleep(stop, self._settings.worker_poll_interval_seconds)
                continue

            token = request_id_ctx.set(f"job-{job.id.hex[:8]}")  # shows up in every log line
            try:
                logger.info(
                    "Started %s job for document %s (attempt %d/%d)",
                    job.job_type.value,
                    job.document_id,
                    job.attempts,
                    job.max_attempts,
                )
                await self._processor.run(job)
            except Exception:
                logger.exception("Unhandled error in job %s", job.id)
            finally:
                request_id_ctx.reset(token)

    async def _maintenance_loop(self, stop: asyncio.Event) -> None:
        last_purge = float("-inf")
        while not stop.is_set():
            try:
                recovered = await recover_stale_jobs(self._session_factory, self._settings)
                if recovered:
                    logger.warning("Recovered %d stale job(s)", recovered)
                if time.monotonic() - last_purge >= PURGE_INTERVAL_SECONDS:
                    tokens, jobs = await purge_old_rows(self._session_factory)
                    last_purge = time.monotonic()
                    if tokens or jobs:
                        logger.info("Purged %d expired token(s) and %d old job(s)", tokens, jobs)
            except Exception:
                logger.exception("Maintenance failed")
            await _sleep(stop, MAINTENANCE_INTERVAL_SECONDS)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        if stop.is_set():
            raise KeyboardInterrupt  # second Ctrl+C: stop waiting and exit now
        logger.info("Signal %s received: finishing current jobs, then stopping", signum)
        loop.call_soon_threadsafe(stop.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    openai_client = build_openai_client(settings)
    qdrant_client = build_qdrant_client(settings)
    vectors = VectorService(qdrant_client, settings)
    try:
        try:
            # A dimension mismatch raises RuntimeError here, so config mistakes fail fast.
            await vectors.ensure_collection()
        except ExternalServiceError:
            logger.warning("Qdrant is not reachable yet; jobs will retry until it is")

        processor = DocumentProcessor(
            AsyncSessionLocal,
            build_storage(settings),
            vectors,
            EmbeddingService(openai_client, settings),
            settings,
        )
        stop = asyncio.Event()
        _install_signal_handlers(stop)
        logger.info("Worker started (concurrency=%d)", settings.worker_concurrency)
        await Worker(settings, AsyncSessionLocal, processor).run(stop)
        logger.info("Worker stopped")
    finally:
        await qdrant_client.close()
        await openai_client.close()
        await engine.dispose()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
