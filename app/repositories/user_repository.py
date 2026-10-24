import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


class UserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self.db.get(User, user_id)

    async def get_by_google_subject(self, subject: str) -> User | None:
        result = await self.db.execute(select(User).where(User.google_subject_id == subject))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(
        self, *, email: str, name: str | None, google_subject_id: str, avatar_url: str | None
    ) -> User:
        user = User(
            email=email, name=name, google_subject_id=google_subject_id, avatar_url=avatar_url
        )
        self.db.add(user)
        await self.db.flush()  # surfaces unique-constraint violations immediately
        return user
