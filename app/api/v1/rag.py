from fastapi import APIRouter

from app.api.deps import CurrentUser, RagSvc
from app.schemas.common import ErrorResponse
from app.schemas.rag import AskRequest, AskResponse

router = APIRouter(tags=["rag"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about your documents",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "document_id is missing, or not yours"},
        409: {"model": ErrorResponse, "description": "The document hasn't finished processing"},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse, "description": "OpenAI or Qdrant failed"},
        503: {"model": ErrorResponse, "description": "AI service busy; see Retry-After"},
    },
)
async def ask(body: AskRequest, user: CurrentUser, service: RagSvc) -> AskResponse:
    """Answers only from your own completed documents. If nothing relevant is found, you get
    `context_found: false` and an empty `sources` list rather than an invented answer."""
    return await service.ask(user, body)
