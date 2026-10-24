from fastapi import APIRouter

from app.api.deps import CurrentUser
from app.models import User
from app.schemas.common import ErrorResponse
from app.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserRead,
    summary="Current user's profile",
    responses={401: {"model": ErrorResponse}},
)
async def read_current_user(user: CurrentUser) -> User:
    return user
