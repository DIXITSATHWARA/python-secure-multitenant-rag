import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Query, Response, UploadFile

from app.api.deps import AppSettings, CurrentUser, DocumentSvc
from app.models import Document, DocumentStatus
from app.schemas.common import ErrorResponse
from app.schemas.document import DocumentPage, DocumentRead, DocumentStatusRead
from app.services.file_validation import read_upload

router = APIRouter(prefix="/documents", tags=["documents"])

UNAUTHORIZED = {401: {"model": ErrorResponse}}
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Missing, or not yours"}}


@router.post(
    "",
    response_model=DocumentRead,
    status_code=202,
    summary="Upload a document",
    responses={
        200: {"model": DocumentRead, "description": "Idempotent replay of an earlier upload"},
        400: {"model": ErrorResponse, "description": "Empty file or content doesn't match type"},
        409: {"model": ErrorResponse, "description": "Idempotency-Key reused with another file"},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse, "description": "File type not allowed"},
        **UNAUTHORIZED,
    },
)
async def upload_document(
    response: Response,
    user: CurrentUser,
    service: DocumentSvc,
    settings: AppSettings,
    file: Annotated[UploadFile, File(description="A PDF or UTF-8 TXT file")],
    name: Annotated[
        str | None, Form(max_length=255, description="Display name (defaults to the filename)")
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
            pattern=r"^[A-Za-z0-9._:-]+$",
            description="Makes retries safe: the same key and file never creates two documents",
        ),
    ] = None,
) -> Document:
    data = await read_upload(file, settings)
    document, created = await service.upload(
        user=user, filename=file.filename, name=name, data=data, idempotency_key=idempotency_key
    )
    if not created:
        response.status_code = 200
    response.headers["Location"] = f"{settings.api_v1_prefix}/documents/{document.id}"
    return document


@router.get(
    "",
    response_model=DocumentPage,
    summary="List your documents (newest first)",
    responses=UNAUTHORIZED,
)
async def list_documents(
    user: CurrentUser,
    service: DocumentSvc,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> DocumentPage:
    documents, total = await service.list(user, limit=limit, offset=offset, status=status_filter)
    return DocumentPage(
        items=[DocumentRead.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get a document's details",
    responses={**UNAUTHORIZED, **NOT_FOUND},
)
async def get_document(document_id: uuid.UUID, user: CurrentUser, service: DocumentSvc) -> Document:
    return await service.get(user, document_id)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusRead,
    summary="Get a document's processing status",
    responses={**UNAUTHORIZED, **NOT_FOUND},
)
async def get_document_status(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentSvc
) -> Document:
    return await service.get(user, document_id)


@router.delete(
    "/{document_id}",
    response_model=DocumentStatusRead,
    status_code=202,
    summary="Delete a document and its vectors",
    responses={**UNAUTHORIZED, **NOT_FOUND},
)
async def delete_document(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentSvc
) -> Document:
    """The document disappears immediately; the background worker (Phase 5) finishes cleanup."""
    return await service.request_delete(user, document_id)
