import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings, get_settings


def production(**overrides) -> Settings:
    """A valid production configuration, with overrides applied and fully re-validated."""
    values = get_settings().model_dump()
    values.update(
        app_env="production",
        cookie_secure=True,
        jwt_secret_key=SecretStr("j" * 40),
        session_secret_key=SecretStr("s" * 40),
        google_client_id="id.apps.googleusercontent.com",
        google_client_secret=SecretStr("secret"),
        google_redirect_uri="https://api.example.com/api/v1/auth/google/callback",
        backend_base_url="https://api.example.com",
        cors_origins=["https://app.example.com"],
        frontend_redirect_url="https://app.example.com/signed-in",
        allowed_hosts=["api.example.com"],
        storage_backend="s3",
        s3_bucket="my-bucket",
    )
    values.update(overrides)
    return Settings.model_validate(values)


def test_a_complete_production_configuration_is_accepted():
    assert production().is_production


@pytest.mark.parametrize("origins", [["http://app.example.com"], ["*"]])
def test_cors_origins_must_be_https(origins):
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        production(cors_origins=origins)


def test_backend_url_must_be_https():
    with pytest.raises(ValidationError, match="BACKEND_BASE_URL"):
        production(backend_base_url="http://api.example.com")


def test_google_redirect_must_be_https():
    with pytest.raises(ValidationError, match="GOOGLE_REDIRECT_URI"):
        production(google_redirect_uri="http://api.example.com/cb")


def test_frontend_redirect_is_required():
    with pytest.raises(ValidationError, match="FRONTEND_REDIRECT_URL"):
        production(frontend_redirect_url="")


@pytest.mark.parametrize("hosts", [[], ["*"]])
def test_allowed_hosts_must_be_real_hostnames(hosts):
    with pytest.raises(ValidationError, match="ALLOWED_HOSTS"):
        production(allowed_hosts=hosts)


def test_local_storage_is_rejected_in_production():
    with pytest.raises(ValidationError, match="STORAGE_BACKEND"):
        production(storage_backend="local")


def test_local_storage_can_be_allowed_explicitly_for_a_single_machine():
    settings = production(storage_backend="local", allow_local_storage_in_production=True)

    assert settings.storage_backend == "local"


def test_s3_requires_a_bucket():
    with pytest.raises(ValidationError, match="S3_BUCKET"):
        production(s3_bucket="")


def test_google_credentials_are_required():
    with pytest.raises(ValidationError, match="GOOGLE_CLIENT_ID"):
        production(google_client_id="")
