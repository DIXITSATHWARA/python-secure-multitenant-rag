from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient

from app.core.body_limit import BodyLimitMiddleware
from app.core.security_headers import SecurityHeadersMiddleware


def client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def body_app(**kwargs) -> FastAPI:
    app = FastAPI()

    @app.post("/small")
    async def small(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.post("/big")
    async def big(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    app.add_middleware(BodyLimitMiddleware, **kwargs)
    return app


async def test_body_within_the_limit_passes():
    async with client(body_app(default_max_bytes=100)) as c:
        resp = await c.post("/small", content=b"x" * 100)

    assert resp.status_code == 200
    assert resp.json() == {"size": 100}


async def test_declared_content_length_over_the_limit_is_rejected():
    async with client(body_app(default_max_bytes=100)) as c:
        resp = await c.post("/small", content=b"x" * 101)

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


async def test_streamed_body_without_content_length_is_capped():
    async def chunks():
        for _ in range(5):
            yield b"x" * 40  # 200 bytes in total, sent chunked

    async with client(body_app(default_max_bytes=100)) as c:
        resp = await c.post("/small", content=chunks())

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


async def test_small_streamed_body_still_works():
    async def chunks():
        for _ in range(3):
            yield b"x" * 20

    async with client(body_app(default_max_bytes=100)) as c:
        resp = await c.post("/small", content=chunks())

    assert resp.status_code == 200
    assert resp.json() == {"size": 60}


async def test_a_path_can_have_its_own_larger_limit():
    app = body_app(default_max_bytes=100, overrides={"/big": 1000})

    async with client(app) as c:
        big = await c.post("/big", content=b"x" * 500)
        small = await c.post("/small", content=b"x" * 500)

    assert big.status_code == 200
    assert small.status_code == 413


def header_app(*, production: bool) -> FastAPI:
    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/cached")
    async def cached(response: Response) -> dict[str, bool]:
        response.headers["Cache-Control"] = "max-age=60"
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware, production=production)
    return app


async def test_default_security_headers_are_set_without_hsts_in_development():
    async with client(header_app(production=False)) as c:
        resp = await c.get("/ok")

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in resp.headers
    assert "content-security-policy" not in resp.headers


async def test_production_adds_hsts_and_a_strict_csp():
    async with client(header_app(production=True)) as c:
        resp = await c.get("/ok")

    assert "max-age=" in resp.headers["strict-transport-security"]
    assert resp.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"


async def test_headers_set_by_the_endpoint_are_not_overridden():
    async with client(header_app(production=False)) as c:
        resp = await c.get("/cached")

    assert resp.headers["cache-control"] == "max-age=60"
    assert resp.headers["x-content-type-options"] == "nosniff"
