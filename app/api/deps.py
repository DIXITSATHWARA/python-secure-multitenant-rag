from pathlib import Path
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService
from app.services.google_oauth import GoogleOAuthService
from app.services.storage import LocalStorage, StorageBackend

DbSession = Annotated[AsyncSession, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]

# auto_error=False so a missing header produces OUR standard 401 error body.
bearer_scheme = HTTPBearer(auto_error=False, description="Access token (JWT)")


def get_google_oauth(settings: AppSettings) -> GoogleOAuthService:
    return GoogleOAuthService(settings)


def get_auth_service(db: DbSession, settings: AppSettings) -> AuthService:
    return AuthService(db, UserRepository(db), RefreshTokenRepository(db), settings)


def get_storage(settings: AppSettings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(Path(settings.local_storage_path))
    raise RuntimeError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")


Storage = Annotated[StorageBackend, Depends(get_storage)]


def get_document_service(db: DbSession, settings: AppSettings, storage: Storage) -> DocumentService:
    return DocumentService(
        db, DocumentRepository(db), ProcessingJobRepository(db), storage, settings
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: DbSession,
    settings: AppSettings,
) -> User:
    """Authenticate the request. Used by every protected route."""
    if credentials is None:
        raise UnauthorizedError("Authentication required.")
    user_id = decode_access_token(credentials.credentials, settings)
    user = await UserRepository(db).get_by_id(user_id)
    # Same message for "no such user" and "disabled": don't reveal which.
    if user is None or not user.is_active:
        raise UnauthorizedError("Invalid or expired access token.")
    return user


GoogleOAuth = Annotated[GoogleOAuthService, Depends(get_google_oauth)]
AuthSvc = Annotated[AuthService, Depends(get_auth_service)]
DocumentSvc = Annotated[DocumentService, Depends(get_document_service)]
CurrentUser = Annotated[User, Depends(get_current_user)]
