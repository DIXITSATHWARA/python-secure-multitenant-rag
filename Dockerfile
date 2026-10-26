FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

# Dependencies pinned by the lock file (used as constraints, so platform-specific
# packages in it are harmless), with the S3 extra for object storage.
COPY pyproject.toml requirements.lock.txt ./
COPY app ./app
RUN pip install -c requirements.lock.txt ".[s3]"

COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

USER appuser
EXPOSE 8000

# One process per container: the rate limiter keeps its counters in memory.
# --no-proxy-headers: client IPs come only from TRUSTED_PROXY_COUNT (see the app's rate limiter).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--no-proxy-headers", "--timeout-keep-alive", "30"]