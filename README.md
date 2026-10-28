# python-secure-multitenant-rag

A secure, multi-tenant Retrieval-Augmented Generation (RAG) backend. Users sign in with Google, upload PDF or TXT documents, and ask questions answered only from their own documents, with citations.

## Features

- **Google sign-in** using the authorization code flow with PKCE, state and nonce, and backend ID-token verification
- **Sessions**: 15-minute JWT access tokens plus rotating, hashed refresh tokens in an HttpOnly cookie, with reuse detection and real logout
- **Document management**: idempotent upload, validation (type, size, content), pagination, status, quota, safe deletion
- **Background processing**: durable PostgreSQL job queue (`SKIP LOCKED`), retries with backoff, crash recovery
- **RAG**: owner-filtered vector search, grounded answers, `[n]` citations, a refusal when evidence is missing, prompt-injection defenses
- **Hardening**: rate limits, body-size caps, security headers, host allow-list, production config validation, log hygiene
- **Operations**: S3-compatible storage, orphan sweep, reindex, account deletion, queue health report

## Tech stack

Python, FastAPI, Pydantic, SQLAlchemy 2 (async), Alembic, PostgreSQL, Qdrant Cloud, OpenAI API, pytest, Ruff.

## Architecture

```
Client -> FastAPI (api -> services -> repositories)
            |-- PostgreSQL: users, documents, refresh tokens, job queue (source of truth)
            |-- Qdrant:     chunk vectors, filtered by user_id on every query
            |-- OpenAI:     embeddings and chat
            `-- Storage:    original files (local disk or S3-compatible)
Worker process: claims jobs, extracts text, chunks, embeds, upserts, verifies, completes
```

Tenant isolation has three layers: a mandatory server-built `user_id` filter in Qdrant, a payload owner re-check in code, and Postgres lookups that include `user_id`. Another user's document returns `404`, identical to a missing one.

## Quick start (Windows / Git Bash)

Requires Python 3.12+, a PostgreSQL database, a Qdrant Cloud cluster, an OpenAI API key, and a Google OAuth client.

```bash
git clone <your-repo-url> rag-app && cd rag-app
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
cp .env.example .env          # then fill in the values

alembic upgrade head
python -m scripts.check_connections
```

Run two processes (both from the project root):

```bash
uvicorn app.main:app --reload        # API on http://localhost:8000  (docs at /docs)
python -m app.workers.runner         # worker: documents stay "pending" without it
```

Sign in by opening `http://localhost:8000/api/v1/auth/google/login` in a browser (use `localhost`, not `127.0.0.1`).

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/auth/google/login` | Start Google sign-in |
| GET | `/api/v1/auth/google/callback` | OAuth callback |
| POST | `/api/v1/auth/refresh` | Rotate refresh token, get a new access token |
| POST | `/api/v1/auth/logout` | Revoke the session |
| GET | `/api/v1/users/me` | Current user |
| POST | `/api/v1/documents` | Upload (supports `Idempotency-Key`) |
| GET | `/api/v1/documents` | List (paginated) |
| GET | `/api/v1/documents/{id}` | Details |
| GET | `/api/v1/documents/{id}/status` | Processing status |
| DELETE | `/api/v1/documents/{id}` | Delete document and vectors |
| POST | `/api/v1/ask` | Ask a question (optional `document_id`) |
| GET | `/api/v1/health`, `/health/ready` | Liveness, readiness |

Errors share one shape: `{"error": {"code", "message", "request_id", "details?"}}`.

## Configuration

All settings come from environment variables (see `.env.example`; `.env.production.example` for production). Production mode (`APP_ENV=production`) refuses to start with unsafe settings such as http URLs, a missing host allow-list, or local storage.

## Testing and quality

```bash
ruff check . && ruff format .
python -m pytest -q
alembic check
```

Tests need a separate database: set `TEST_DATABASE_URL` to one whose name contains `test`. OpenAI is faked and Qdrant runs in memory, so no cloud calls are made. An authorization sweep fails any route that forgets authentication.

## Operations

| Task | Command |
|---|---|
| Queue health (non-zero exit = needs attention) | `python -m scripts.ops_report` |
| Retry failed jobs | `python -m scripts.requeue_jobs` |
| Orphan files and vectors (dry run, `--apply` deletes) | `python -m scripts.sweep_orphans` |
| Re-embed all documents | `python -m scripts.reindex_documents --apply` |
| Delete an account | `python -m scripts.delete_user EMAIL`, then `--finalize` |
| Measure retrieval quality | `python -m scripts.eval_retrieval EMAIL cases.json` |

A `Dockerfile` is included. Run one uvicorn process per instance, because rate limits are held in memory.

## Project structure

```
app/
  api/v1/        route handlers (thin)
  core/          config, security, logging, middleware, errors
  models/        SQLAlchemy models
  schemas/       Pydantic request and response models
  repositories/  database queries only
  services/      business logic, OpenAI, Qdrant, storage
  workers/       job runner, processor, maintenance, sweep
alembic/         migrations
scripts/         operator tools
tests/
```

## Known limitations

Stateless questions (no follow-ups), no streaming, vector-only retrieval (exact identifiers can be missed), no OCR for scanned PDFs, and generic document-level questions ("what is the name?") can miss because no single chunk represents the whole document.

## Roadmap

Conversation history, streaming answers, hybrid search and reranking, DOCX support, a header chunk per document, Redis-backed rate limits when running multiple replicas, and metrics and tracing.
