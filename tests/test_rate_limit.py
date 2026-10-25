import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.rate_limit import (
    IpRateLimitMiddleware,
    IpRule,
    Limit,
    SlidingWindowLimiter,
    client_ip,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def limiter(clock: Clock) -> SlidingWindowLimiter:
    return SlidingWindowLimiter(clock=clock)


PER_MINUTE = (Limit(3, 60),)


# ---- the limiter --------------------------------------------------------------------


def test_allows_up_to_the_limit_then_blocks(limiter):
    for _ in range(3):
        assert limiter.check("k", PER_MINUTE).allowed

    decision = limiter.check("k", PER_MINUTE)

    assert not decision.allowed
    assert decision.retry_after == 60


def test_blocked_requests_are_not_recorded(limiter, clock):
    for _ in range(3):
        limiter.check("k", PER_MINUTE)
    clock.advance(30)
    for _ in range(5):
        assert not limiter.check("k", PER_MINUTE).allowed  # hammering must not extend the ban

    clock.advance(31)  # the original three hits are now 61 s old

    assert limiter.check("k", PER_MINUTE).allowed


def test_the_window_slides(limiter, clock):
    limiter.check("k", PER_MINUTE)  # t=1000
    clock.advance(20)
    limiter.check("k", PER_MINUTE)  # t=1020
    clock.advance(20)
    limiter.check("k", PER_MINUTE)  # t=1040
    clock.advance(10)  # t=1050: three hits inside the window

    blocked = limiter.check("k", PER_MINUTE)

    assert not blocked.allowed
    assert blocked.retry_after == 10  # the t=1000 hit leaves the window at 1060
    clock.advance(10)
    assert limiter.check("k", PER_MINUTE).allowed


def test_every_window_is_enforced_and_retry_after_is_the_longest_wait(limiter, clock):
    limits = (Limit(5, 60), Limit(6, 3600))
    for _ in range(5):
        assert limiter.check("k", limits).allowed
    clock.advance(61)  # the per-minute window is clear again
    assert limiter.check("k", limits).allowed  # 6th hit of the hour

    decision = limiter.check("k", limits)

    assert not decision.allowed  # blocked by the hourly window, not the minute one
    assert decision.retry_after == 3539  # first hit (t=1000) expires at 4600; now is 1061


def test_keys_are_independent(limiter):
    for _ in range(3):
        limiter.check("a", PER_MINUTE)

    assert not limiter.check("a", PER_MINUTE).allowed
    assert limiter.check("b", PER_MINUTE).allowed


def test_non_positive_limits_are_ignored(limiter):
    for _ in range(50):
        assert limiter.check("k", (Limit(0, 60),)).allowed
        assert limiter.check("k", ()).allowed


def test_idle_keys_are_swept(limiter, clock):
    limiter.check("a", PER_MINUTE)
    limiter.check("b", PER_MINUTE)
    clock.advance(200)

    limiter.check("c", PER_MINUTE)  # the sweep runs on the next call after 60 s

    assert limiter.key_count == 1


def test_memory_is_bounded_by_max_keys(clock):
    small = SlidingWindowLimiter(clock=clock, max_keys=2)

    for key in ("a", "b", "c", "d"):
        small.check(key, PER_MINUTE)

    assert small.key_count == 2


# ---- client address -----------------------------------------------------------------

PEER = "203.0.113.9"


def make_scope(peer: str | None = PEER, forwarded: str | None = None) -> dict:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    return {"type": "http", "client": (peer, 5000) if peer else None, "headers": headers}


def test_forwarded_header_is_ignored_when_no_proxy_is_trusted():
    assert client_ip(make_scope(forwarded="1.2.3.4"), trusted_proxies=0) == PEER


def test_uses_the_entry_added_by_the_trusted_proxy_not_a_spoofed_one():
    scope = make_scope(forwarded="6.6.6.6, 1.2.3.4")

    assert client_ip(scope, trusted_proxies=1) == "1.2.3.4"


def test_two_trusted_proxies_use_the_second_entry_from_the_right():
    scope = make_scope(forwarded="6.6.6.6, 1.2.3.4, 10.0.0.1")

    assert client_ip(scope, trusted_proxies=2) == "1.2.3.4"


def test_falls_back_to_the_peer_when_the_header_is_too_short():
    assert client_ip(make_scope(), trusted_proxies=1) == PEER
    assert client_ip(make_scope(forwarded="1.2.3.4"), trusted_proxies=2) == PEER


def test_garbage_in_the_forwarded_header_falls_back_to_the_peer():
    assert client_ip(make_scope(forwarded="not-an-ip"), trusted_proxies=1) == PEER


def test_ipv6_clients_are_grouped_by_their_64_bit_prefix():
    first = client_ip(make_scope(peer="2001:db8:1:2:aaaa:bbbb:cccc:dddd"))
    second = client_ip(make_scope(peer="2001:db8:1:2:1111:2222:3333:4444"))

    assert first == second == "2001:db8:1:2::/64"


def test_ipv4_mapped_ipv6_addresses_are_unwrapped():
    assert client_ip(make_scope(peer="::ffff:1.2.3.4")) == "1.2.3.4"


def test_missing_client_is_unknown():
    assert client_ip(make_scope(peer=None)) == "unknown"


# ---- the per-IP middleware ------------------------------------------------------------


def make_app(rules: list[IpRule], trusted: int = 0) -> FastAPI:
    app = FastAPI()

    @app.get("/login")
    async def login() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/free")
    async def free() -> dict[str, bool]:
        return {"ok": True}

    @app.api_route("/upload", methods=["GET", "POST"])
    async def upload() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/auth/a")
    async def auth_a() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/auth/b")
    async def auth_b() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(
        IpRateLimitMiddleware,
        limiter=SlidingWindowLimiter(clock=Clock()),
        rules=rules,
        trusted_proxies=trusted,
    )
    return app


def client_for(app: FastAPI, host: str = "198.51.100.1") -> AsyncClient:
    transport = ASGITransport(app=app, client=(host, 1234))
    return AsyncClient(transport=transport, base_url="http://t")


LOGIN_RULE = IpRule("login", "/login", (Limit(2, 60),))


async def test_blocked_requests_get_the_standard_error_and_retry_after():
    app = make_app([LOGIN_RULE])

    async with client_for(app) as c:
        codes = [(await c.get("/login")).status_code for _ in range(2)]
        blocked = await c.get("/login")

    assert codes == [200, 200]
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "60"
    assert blocked.json()["error"]["code"] == "rate_limited"


async def test_other_paths_and_other_addresses_are_not_affected():
    app = make_app([LOGIN_RULE])

    async with client_for(app) as first, client_for(app, "198.51.100.2") as second:
        for _ in range(3):
            await first.get("/login")
        assert (await first.get("/free")).status_code == 200
        assert (await second.get("/login")).status_code == 200


async def test_a_rule_only_applies_to_its_listed_methods():
    rule = IpRule("upload", "/upload", (Limit(1, 60),), methods=frozenset({"POST"}))
    app = make_app([rule])

    async with client_for(app) as c:
        gets = [(await c.get("/upload")).status_code for _ in range(3)]
        first_post = await c.post("/upload")
        second_post = await c.post("/upload")

    assert gets == [200, 200, 200]
    assert first_post.status_code == 200
    assert second_post.status_code == 429


async def test_prefix_rules_share_one_bucket():
    app = make_app([IpRule("auth", "/auth/", (Limit(1, 60),), prefix=True)])

    async with client_for(app) as c:
        assert (await c.get("/auth/a")).status_code == 200
        assert (await c.get("/auth/b")).status_code == 429


async def test_preflight_requests_do_not_consume_the_budget():
    app = make_app([LOGIN_RULE])

    async with client_for(app) as c:
        for _ in range(5):
            await c.options("/login")
        assert (await c.get("/login")).status_code == 200
        assert (await c.get("/login")).status_code == 200


async def test_trusted_proxy_header_separates_clients_and_resists_spoofing():
    app = make_app([LOGIN_RULE], trusted=1)
    real = {"X-Forwarded-For": "1.1.1.1"}
    spoofed = {"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}  # leading entry is client-controlled
    other = {"X-Forwarded-For": "2.2.2.2"}

    async with client_for(app) as c:
        assert (await c.get("/login", headers=real)).status_code == 200
        assert (await c.get("/login", headers=spoofed)).status_code == 200
        assert (await c.get("/login", headers=real)).status_code == 429
        assert (await c.get("/login", headers=other)).status_code == 200
