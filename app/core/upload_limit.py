from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.schemas.common import ErrorBody, ErrorResponse

# Multipart framing and form fields add a little on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class UploadSizeLimitMiddleware:
    """Rejects an upload whose declared Content-Length is too big, before the body is parsed.

    Without this, the framework would spool the entire multipart body to disk first.
    Clients using chunked encoding send no Content-Length, so the streamed check in
    read_upload() and a reverse-proxy limit (Phase 8) back this up.
    """

    def __init__(self, app: ASGIApp, *, path: str, max_bytes: int) -> None:
        self.app = app
        self.path = path
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == self.path:
            declared = Headers(scope=scope).get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
                body = ErrorResponse(
                    error=ErrorBody(
                        code="payload_too_large",
                        message="The uploaded file is too large.",
                        request_id=scope.get("state", {}).get("request_id"),
                    )
                )
                response = JSONResponse(
                    body.model_dump(mode="json", exclude_none=True), status_code=413
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
