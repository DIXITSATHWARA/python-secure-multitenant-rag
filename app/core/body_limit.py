from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.schemas.common import ErrorBody, ErrorResponse

TOO_LARGE = "The request body is too large."


class BodyLimitMiddleware:
    """Caps request bodies by counting the bytes as they stream in.

    FastAPI reads a whole JSON body into memory before validating it, and a client using
    chunked encoding sends no Content-Length at all. So the declared length is checked first
    (cheap), and the stream is counted as well (catches everything else)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_max_bytes: int,
        overrides: Mapping[str, int] | None = None,
    ) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.overrides = dict(overrides or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.overrides.get(scope["path"], self.default_max_bytes)
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(scope, receive, send)
            return

        received = 0
        rejected = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    if not started:
                        await self._reject(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return  # the 413 is already on the wire; drop whatever the app tries next
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # After a rejection the app sees a "disconnect" and may raise. That is expected.
            if not rejected:
                raise

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        body = ErrorResponse(
            error=ErrorBody(
                code="payload_too_large",
                message=TOO_LARGE,
                request_id=scope.get("state", {}).get("request_id"),
            )
        )
        response = JSONResponse(
            body.model_dump(mode="json", exclude_none=True),
            status_code=413,
            headers={"Connection": "close"},  # the unread body makes keep-alive unsafe
        )
        await response(scope, receive, send)
