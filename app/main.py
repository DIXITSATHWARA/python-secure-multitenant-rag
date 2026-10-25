import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.body_limit import BodyLimitMiddleware
from app.core.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.exceptions import ExternalServiceError
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.rate_limit import IpRateLimitMiddleware, SlidingWindowLimiter, build_ip_rules
from app.core.security_headers import SecurityHeadersMiddleware
from app.core.upload_limit import MULTIPART_OVERHEAD_BYTES, UploadSizeLimitMiddleware
from app.db.session import engine
from app.services.clients import build_openai_client, build_qdrant_client
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)

    openai_client = build_openai_client(settings)
    app.state.openai = openai_client
    qdrant = build_qdrant_client(settings)
    vectors = VectorService(qdrant, settings)
    app.state.vectors = vectors
    try:
        # Creates the collection only if it is missing. A dimension mismatch (RuntimeError)
        # is a configuration bug, so it deliberately stops startup.
        await vectors.ensure_collection()
    except ExternalServiceError:
        logger.warning("Qdrant is not reachable at startup; /health/ready will report it")

    yield

    await openai_client.close()
    await qdrant.close()
    await engine.dispose()  # close pooled DB connections cleanly
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    # Interactive docs are convenient locally; hide them in production.
    docs_enabled = not settings.is_production
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # Created here (not in the lifespan) so every app instance has its own limiter.
    limiter = SlidingWindowLimiter()
    app.state.limiter = limiter

    upload_path = f"{settings.api_v1_prefix}/documents"
    upload_max = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    # add_middleware wraps: the LAST one added is the outermost. Request path, outer to inner:
    # RequestContext > SecurityHeaders > CORS > IpRateLimit > BodyLimit > UploadSizeLimit > app
    app.add_middleware(UploadSizeLimitMiddleware, path=upload_path, max_bytes=upload_max)
    app.add_middleware(
        BodyLimitMiddleware,
        default_max_bytes=settings.max_request_body_kb * 1024,
        overrides={upload_path: upload_max},
    )
    app.add_middleware(
        IpRateLimitMiddleware,
        limiter=limiter,
        rules=build_ip_rules(settings) if settings.rate_limit_enabled else [],
        trusted_proxies=settings.trusted_proxy_count,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # explicit list, never "*" with credentials
        allow_credentials=True,  # needed for the HttpOnly refresh-token cookie
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        expose_headers=["X-Request-ID", "Location", "Retry-After"],
    )
    app.add_middleware(SecurityHeadersMiddleware, production=settings.is_production)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
