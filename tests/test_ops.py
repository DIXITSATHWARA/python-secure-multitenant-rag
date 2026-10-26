from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from app.models import Document, DocumentStatus, JobStatus, ProcessingJob
from app.workers.ops import OpsStats, collect_ops_stats

LEASE = 900


async def stats(session_factory, now) -> OpsStats:
    return await collect_ops_stats(session_factory, lease_seconds=LEASE, now=now)


async def set_jobs(session_factory, **values) -> None:
    async with session_factory() as db:
        await db.execute(update(ProcessingJob).values(**values))
        await db.commit()


async def test_an_empty_system_is_healthy(session_factory):
    result = await stats(session_factory, datetime.now(UTC))

    assert result == OpsStats(0, 0, 0, None, 0, 0)
    assert result.problems() == []


async def test_reports_queue_depth_and_the_age_of_the_oldest_runnable_job(
    upload, alice, session_factory
):
    now = datetime.now(UTC)
    await upload(alice)
    await set_jobs(session_factory, run_after=now - timedelta(seconds=120))

    result = await stats(session_factory, now)

    assert result.queued == 1
    assert result.oldest_due_age_seconds == 120.0
    assert result.problems() == []  # 120 s is under the default 600 s threshold


async def test_jobs_waiting_out_a_backoff_are_not_counted_as_overdue(
    upload, alice, session_factory
):
    now = datetime.now(UTC)
    await upload(alice)
    await set_jobs(session_factory, run_after=now + timedelta(seconds=300))

    result = await stats(session_factory, now)

    assert result.queued == 1
    assert result.oldest_due_age_seconds is None


async def test_a_long_wait_is_a_problem_when_it_exceeds_the_threshold(
    upload, alice, session_factory
):
    now = datetime.now(UTC)
    await upload(alice)
    await set_jobs(session_factory, run_after=now - timedelta(seconds=120))

    problems = (await stats(session_factory, now)).problems(max_queue_age_seconds=60)

    assert len(problems) == 1
    assert "worker" in problems[0]


async def test_failed_jobs_are_a_problem(upload, alice, session_factory):
    await upload(alice)
    await set_jobs(session_factory, status=JobStatus.FAILED)

    result = await stats(session_factory, datetime.now(UTC))

    assert result.failed == 1
    assert any("failed" in problem for problem in result.problems())


async def test_running_jobs_with_an_expired_lease_are_a_problem(upload, alice, session_factory):
    now = datetime.now(UTC)
    await upload(alice)
    await set_jobs(session_factory, status=JobStatus.RUNNING, locked_at=now - timedelta(hours=2))

    result = await stats(session_factory, now)

    assert (result.running, result.stale_running) == (1, 1)
    assert any("lease" in problem for problem in result.problems())


async def test_documents_stuck_in_deleting_are_a_problem(upload, alice, session_factory):
    now = datetime.now(UTC)
    doc_id = (await upload(alice)).json()["id"]
    async with session_factory() as db:
        await db.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(status=DocumentStatus.DELETING, updated_at=now - timedelta(hours=2))
        )
        await db.commit()

    result = await stats(session_factory, now)

    assert result.stuck_deleting == 1
    assert any("deleting" in problem for problem in result.problems())
