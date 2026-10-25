import asyncio
import logging

from fastapi import APIRouter, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, OptionalVectors
from app.schemas.health import HealthResponse, ReadinessResponse
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Is the process up? Touches no dependencies, so it never flaps."""
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    summary="Readiness probe",
)
async def readiness(
    db: DbSession, vectors: OptionalVectors, response: Response
) -> ReadinessResponse:
    """Can this instance serve traffic? Checks PostgreSQL and Qdrant. OpenAI is deliberately
    not checked: a third-party outage shouldn't take your app out of rotation."""
    checks = {
        "postgres": await _check_postgres(db),
        "qdrant": await _check_qdrant(vectors),
    }
    healthy = all(value == "ok" for value in checks.values())
    if not healthy:
        response.status_code = 503
    return ReadinessResponse(status="ok" if healthy else "degraded", checks=checks)


async def _check_postgres(db: AsyncSession) -> str:
    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=3)
    except Exception:
        # Log details server-side; never leak connection errors to clients.
        logger.warning("PostgreSQL readiness check failed", exc_info=True)
        return "unavailable"
    return "ok"


async def _check_qdrant(vectors: VectorService | None) -> str:
    if vectors is None:
        return "unavailable"
    return "ok" if await vectors.ping() else "unavailable"
