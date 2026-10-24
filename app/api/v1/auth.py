from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.deps import AppSettings, AuthSvc, GoogleOAuth
from app.core.config import Settings
from app.core.exceptions import BadRequestError, ForbiddenError, UnauthorizedError
from app.schemas.auth import TokenResponse
from app.schemas.common import ErrorResponse
from app.services.auth_service import IssuedTokens

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
STATE_COOKIE = "oauth_state"

QueryParam = Annotated[str | None, Query(max_length=2048)]
NO_STORE = {"Cache-Control": "no-store"}


# ---- cookie helpers -----------------------------------------------------------------


def _refresh_path(settings: Settings) -> str:
    # The refresh cookie is only ever sent to /auth/* endpoints, never to document APIs.
    return f"{settings.api_v1_prefix}/auth"


def _state_path(settings: Settings) -> str:
    return f"{settings.api_v1_prefix}/auth/google"


def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        token,
        max_age=settings.refresh_token_expire_days * 86400,
        httponly=True,  # invisible to JavaScript, which blunts XSS token theft
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=_refresh_path(settings),
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        REFRESH_COOKIE,
        path=_refresh_path(settings),
        secure=settings.cookie_secure,
        httponly=True,
        samesite=settings.cookie_samesite,
    )


def _clear_refresh_cookie_header(settings: Settings) -> str:
    scratch = Response()
    _clear_refresh_cookie(scratch, settings)
    return scratch.headers["set-cookie"]


def _check_origin(request: Request, settings: Settings) -> None:
    """CSRF defense in depth for cookie-authenticated POSTs. Browsers always send Origin
    on cross-origin POSTs; non-browser clients (curl) send none and are allowed."""
    origin = request.headers.get("origin")
    if origin is None:
        return
    allowed = {*settings.cors_origins, settings.backend_base_url.rstrip("/")}
    if origin not in allowed:
        raise ForbiddenError("Origin not allowed.")


def _tokens_json(tokens: IssuedTokens) -> JSONResponse:
    body = TokenResponse(access_token=tokens.access_token, expires_in=tokens.expires_in)
    return JSONResponse(body.model_dump(), headers=NO_STORE)


# ---- routes -------------------------------------------------------------------------


@router.get(
    "/google/login",
    status_code=302,
    summary="Start Google sign-in",
    responses={302: {"description": "Redirect to Google"}, 503: {"model": ErrorResponse}},
)
async def google_login(google: GoogleOAuth, settings: AppSettings) -> RedirectResponse:
    """Open this URL in a browser tab. (Swagger's "Try it out" can't follow the redirect.)"""
    url, state_cookie = google.begin_login()
    response = RedirectResponse(url, status_code=302, headers=NO_STORE)
    response.set_cookie(
        STATE_COOKIE,
        state_cookie,
        max_age=settings.oauth_state_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",  # must be Lax: Google's redirect back is a cross-site navigation
        path=_state_path(settings),
    )
    return response


@router.get(
    "/google/callback",
    summary="Google OAuth callback",
    responses={
        200: {"model": TokenResponse, "description": "Dev mode (no FRONTEND_REDIRECT_URL)"},
        302: {"description": "Redirect to the frontend"},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def google_callback(
    request: Request,
    google: GoogleOAuth,
    auth: AuthSvc,
    settings: AppSettings,
    code: QueryParam = None,
    state: QueryParam = None,
    error: QueryParam = None,
) -> Response:
    if error:
        # Never echo Google's "error" value back to the client.
        raise BadRequestError("Google sign-in was cancelled or denied.")
    if not code or not state:
        raise BadRequestError("Missing authorization code or state.")

    identity = await google.complete_login(
        code=code, state=state, cookie_value=request.cookies.get(STATE_COOKIE)
    )
    tokens = await auth.sign_in_with_google(identity)

    if settings.frontend_redirect_url:
        response: Response = RedirectResponse(
            settings.frontend_redirect_url, status_code=302, headers=NO_STORE
        )
    else:
        response = _tokens_json(tokens)
    _set_refresh_cookie(response, tokens.refresh_token, settings)
    # The flow cookie is single-use.
    response.delete_cookie(
        STATE_COOKIE,
        path=_state_path(settings),
        secure=settings.cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return response


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate the refresh token and get a new access token",
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
async def refresh(request: Request, auth: AuthSvc, settings: AppSettings) -> JSONResponse:
    """Reads the refresh token from its HttpOnly cookie. No request body."""
    _check_origin(request, settings)
    try:
        tokens = await auth.refresh(request.cookies.get(REFRESH_COOKIE))
    except UnauthorizedError as exc:
        # Tell the browser to drop the dead cookie.
        exc.headers = {**(exc.headers or {}), "Set-Cookie": _clear_refresh_cookie_header(settings)}
        raise
    response = _tokens_json(tokens)
    _set_refresh_cookie(response, tokens.refresh_token, settings)
    return response


@router.post(
    "/logout",
    status_code=204,
    summary="Revoke this session",
    responses={403: {"model": ErrorResponse}},
)
async def logout(request: Request, auth: AuthSvc, settings: AppSettings) -> Response:
    _check_origin(request, settings)
    await auth.logout(request.cookies.get(REFRESH_COOKIE))
    response = Response(status_code=204)
    _clear_refresh_cookie(response, settings)
    return response
