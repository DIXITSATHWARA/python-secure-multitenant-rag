import logging
from typing import Annotated

from fastapi import Depends, Request

from app.api.deps import AppSettings, CurrentUser
from app.core.config import Settings
from app.core.exceptions import RateLimitedError
from app.core.rate_limit import Limit, SlidingWindowLimiter
from app.models import User

logger = logging.getLogger(__name__)


def get_limiter(request: Request) -> SlidingWindowLimiter:
    """One limiter per app instance, created in create_app()."""
    return request.app.state.limiter


Limiter = Annotated[SlidingWindowLimiter, Depends(get_limiter)]


def _enforce(
    limiter: SlidingWindowLimiter,
    settings: Settings,
    user: User,
    bucket: str,
    limits: tuple[Limit, ...],
) -> None:
    if not settings.rate_limit_enabled:
        return
    decision = limiter.check(f"{bucket}:user:{user.id}", limits)
    if decision.allowed:
        return
    logger.warning("Rate limit hit: rule=%s user=%s", bucket, user.id)
    raise RateLimitedError(headers={"Retry-After": str(decision.retry_after)})


async def limit_ask(user: CurrentUser, settings: AppSettings, limiter: Limiter) -> None:
    _enforce(
        limiter,
        settings,
        user,
        "ask",
        (
            Limit(settings.rate_limit_ask_per_minute, 60),
            Limit(settings.rate_limit_ask_per_hour, 3600),
        ),
    )


async def limit_upload(user: CurrentUser, settings: AppSettings, limiter: Limiter) -> None:
    _enforce(
        limiter,
        settings,
        user,
        "upload",
        (
            Limit(settings.rate_limit_upload_per_minute, 60),
            Limit(settings.rate_limit_upload_per_hour, 3600),
        ),
    )
