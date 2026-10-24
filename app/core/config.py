from functools import lru_cache
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    app_name: str = "rag-app"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    backend_base_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # PostgreSQL
    database_url: str
    test_database_url: str = ""
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Qdrant
    qdrant_url: str
    qdrant_api_key: SecretStr
    qdrant_collection: str = "document_chunks"

    # OpenAI
    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_chat_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 30.0

    # Auth
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 14
    refresh_reuse_grace_seconds: int = 10
    session_secret_key: SecretStr
    google_client_id: str = ""
    google_client_secret: SecretStr = SecretStr("")
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"
    oauth_state_ttl_seconds: int = 600
    # Where to send the browser after a successful sign-in. Empty = return JSON (dev only).
    frontend_redirect_url: str = ""
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"

    # Documents
    max_upload_mb: int = 20
    allowed_file_types: list[str] = ["pdf", "txt"]
    storage_backend: str = "local"
    local_storage_path: str = "./storage"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.30
    max_context_chars: int = 8000

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @model_validator(mode="after")
    def _validate_security_settings(self) -> Self:
        jwt_key = self.jwt_secret_key.get_secret_value()
        session_key = self.session_secret_key.get_secret_value()
        if len(jwt_key) < 32 or len(session_key) < 32:
            raise ValueError("JWT_SECRET_KEY and SESSION_SECRET_KEY must be at least 32 characters")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("COOKIE_SAMESITE=none requires COOKIE_SECURE=true")
        if self.is_production:
            if not self.cookie_secure:
                raise ValueError("COOKIE_SECURE must be true in production")
            if jwt_key == session_key:
                raise ValueError("JWT_SECRET_KEY and SESSION_SECRET_KEY must differ")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
