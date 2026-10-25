import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.exceptions import ExternalServiceError
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.upload_limit import MULTIPART_OVERHEAD_BYTES, UploadSizeLimitMiddleware
from app.db.session import engine
from app.services.clients import build_qdrant_client
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)

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

    # Middleware added later wraps earlier ones: upload limit is innermost, then CORS
    # (so 413 responses carry CORS headers), then RequestContext (outermost).
    app.add_middleware(
        UploadSizeLimitMiddleware,
        path=f"{settings.api_v1_prefix}/documents",
        max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # explicit list, never "*" with credentials
        allow_credentials=True,  # needed for the HttpOnly refresh-token cookie
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        expose_headers=["X-Request-ID", "Location"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
