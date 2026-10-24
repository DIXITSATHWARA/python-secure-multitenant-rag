from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()
api_router.include_router(health.router)
# Phase 3+: auth, users, documents, rag routers are added here.
