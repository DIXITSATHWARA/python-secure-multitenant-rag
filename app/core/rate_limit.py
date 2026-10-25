"""In-memory sliding-window rate limiting.

State lives in this process, which is right for a single API instance. With several
instances (or several uvicorn workers) each keeps its own counters, so the effective limit
multiplies. That is the point where a shared store such as Redis starts to pay for itself.
"""

import ipaddress
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings
from app.schemas.common import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)

RATE_LIMITED_MESSAGE = "Too many requests. Please try again later."


@dataclass(frozen=True)
class Limit:
    """At most `count` requests per `window_seconds`. A count of 0 disables the limit."""

    count: int
    window_seconds: int


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: int = 0  # whole seconds to wait; only meaningful when not allowed


class SlidingWindowLimiter:
    SWEEP_INTERVAL_SECONDS = 60

    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic, max_keys: int = 100_000
    ) -> None:
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._longest_window = 1
        self._last_sweep = clock()

    @property
    def key_count(self) -> int:
        return len(self._hits)

    def check(self, key: str, limits: Sequence[Limit]) -> Decision:
        """Record one hit and allow it, or refuse it without recording anything.

        Synchronous with no awaits, so it is atomic on the event loop."""
        active = [lim for lim in limits if lim.count > 0 and lim.window_seconds > 0]
        if not active:
            return Decision(True)
        now = self._clock()
        self._maybe_sweep(now)
        longest = max(lim.window_seconds for lim in active)
        self._longest_window = max(self._longest_window, longest)

        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
        while hits and hits[0] <= now - longest:
            hits.popleft()

        wait = 0.0
        for lim in active:
            recent = [t for t in hits if t > now - lim.window_seconds]
            if len(recent) >= lim.count:
                # The request fits again once enough old hits have left this window.
                frees_at = recent[len(recent) - lim.count] + lim.window_seconds
                wait = max(wait, frees_at - now)
        if wait > 0:
            return Decision(False, max(1, math.ceil(wait)))

        if key not in self._hits and len(self._hits) >= self._max_keys:
            del self._hits[next(iter(self._hits))]  # evict the oldest key: memory stays bounded
        hits.append(now)
        self._hits[key] = hits
        return Decision(True)

    def _maybe_sweep(self, now: float) -> None:
        if now - self._last_sweep < self.SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        cutoff = now - self._longest_window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


# ---- client address -----------------------------------------------------------------


def normalize_ip(value: str) -> str | None:
    """Canonical limiter key. IPv6 clients are grouped by /64 (one subscriber)."""
    try:
        ip = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_ip(scope: Scope, trusted_proxies: int = 0) -> str:
    """The caller's address. X-Forwarded-For is only believed when TRUSTED_PROXY_COUNT > 0.

    With N trusted proxies in front, each appends the address it saw, so the real client is
    the Nth entry from the right. Anything further left is client-supplied and spoofable."""
    client = scope.get("client")
    peer = normalize_ip(client[0]) if client else None
    if trusted_proxies > 0:
        forwarded = ",".join(Headers(scope=scope).getlist("x-forwarded-for"))
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if len(parts) >= trusted_proxies:
            claimed = normalize_ip(parts[-trusted_proxies])
            if claimed is not None:
                return claimed
    return peer or "unknown"


# ---- per-IP middleware ---------------------------------------------------------------


@dataclass(frozen=True)
class IpRule:
    name: str
    path: str
    limits: tuple[Limit, ...]
    methods: frozenset[str] = frozenset()  # empty = any method
    prefix: bool = False  # True = `path` is a prefix, False = exact match

    def matches(self, method: str, path: str) -> bool:
        if self.methods and method not in self.methods:
            return False
        return path.startswith(self.path) if self.prefix else path == self.path


def build_ip_rules(settings: Settings) -> list[IpRule]:
    prefix = settings.api_v1_prefix
    return [
        IpRule(
            "auth",
            f"{prefix}/auth/",
            (Limit(settings.rate_limit_auth_per_minute, 60),),
            prefix=True,
        ),
        # A guard that runs before the multipart body is parsed. The per-user limit can only
        # run after authentication, which is after the body has been read.
        IpRule(
            "upload",
            f"{prefix}/documents",
            (Limit(settings.rate_limit_upload_ip_per_minute, 60),),
            methods=frozenset({"POST"}),
        ),
    ]


class IpRateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: SlidingWindowLimiter,
        rules: Sequence[IpRule],
        trusted_proxies: int = 0,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.rules = list(rules)
        self.trusted_proxies = trusted_proxies

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # OPTIONS (CORS preflight) never counts against the budget.
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        rule = next((r for r in self.rules if r.matches(scope["method"], scope["path"])), None)
        if rule is None:
            await self.app(scope, receive, send)
            return

        ip = client_ip(scope, self.trusted_proxies)
        decision = self.limiter.check(f"{rule.name}:ip:{ip}", rule.limits)
        if decision.allowed:
            await self.app(scope, receive, send)
            return

        logger.warning("Rate limit hit: rule=%s ip=%s", rule.name, ip)
        body = ErrorResponse(
            error=ErrorBody(
                code="rate_limited",
                message=RATE_LIMITED_MESSAGE,
                request_id=scope.get("state", {}).get("request_id"),
            )
        )
        response = JSONResponse(
            body.model_dump(mode="json", exclude_none=True),
            status_code=429,
            headers={"Retry-After": str(decision.retry_after)},
        )
        await response(scope, receive, send)
