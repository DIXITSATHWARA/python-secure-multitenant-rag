import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ConflictError, ForbiddenError, UnauthorizedError
from app.core.security import create_access_token, generate_refresh_token, hash_token
from app.models import User
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.services.google_oauth import GoogleIdentity

logger = logging.getLogger(__name__)

INVALID_REFRESH = "Invalid or expired refresh token."


@dataclass(frozen=True)
class IssuedTokens:
    access_token: str
    expires_in: int
    refresh_token: str


class AuthService:
    def __init__(
        self,
        db: AsyncSession,
        users: UserRepository,
        tokens: RefreshTokenRepository,
        settings: Settings,
    ) -> None:
        self.db = db
        self.users = users
        self.tokens = tokens
        self.settings = settings

    async def sign_in_with_google(self, identity: GoogleIdentity) -> IssuedTokens:
        user = await self._get_or_create_user(identity)
        if not user.is_active:
            raise ForbiddenError("This account has been disabled.")
        issued = await self._issue_tokens(user.id, family_id=uuid.uuid4())
        await self.db.commit()
        return issued

    async def refresh(self, raw_token: str | None) -> IssuedTokens:
        """Rotate: the presented token is revoked and a new one issued in the same family."""
        if not raw_token:
            raise UnauthorizedError("Missing refresh token.")
        now = datetime.now(UTC)
        stored = await self.tokens.get_by_hash(hash_token(raw_token))  # row-locked
        if stored is None:
            raise UnauthorizedError(INVALID_REFRESH)

        if stored.revoked_at is not None:
            grace = timedelta(seconds=self.settings.refresh_reuse_grace_seconds)
            if now - stored.revoked_at >= grace:
                # A rotated-out token came back late: assume theft, kill the whole session.
                logger.warning(
                    "Refresh token reuse detected; revoking family %s (user %s)",
                    stored.family_id,
                    stored.user_id,
                )
                await self.tokens.revoke_family(stored.family_id, now)
                await self.db.commit()
            # Inside the grace window it's most likely two tabs racing: just refuse this one.
            raise UnauthorizedError(INVALID_REFRESH)

        if stored.expires_at <= now:
            raise UnauthorizedError(INVALID_REFRESH)

        user = await self.users.get_by_id(stored.user_id)
        if user is None or not user.is_active:
            await self.tokens.revoke_family(stored.family_id, now)
            await self.db.commit()
            raise UnauthorizedError(INVALID_REFRESH)

        stored.revoked_at = now
        issued = await self._issue_tokens(user.id, family_id=stored.family_id)
        await self.db.commit()
        return issued

    async def logout(self, raw_token: str | None) -> None:
        """Idempotent: unknown or missing tokens are silently ignored."""
        if not raw_token:
            return
        stored = await self.tokens.get_by_hash(hash_token(raw_token))
        if stored is None:
            return
        await self.tokens.revoke_family(stored.family_id, datetime.now(UTC))
        await self.db.commit()

    # ---- internals --------------------------------------------------------------------

    async def _issue_tokens(self, user_id: uuid.UUID, *, family_id: uuid.UUID) -> IssuedTokens:
        raw_refresh = generate_refresh_token()
        await self.tokens.create(
            user_id=user_id,
            family_id=family_id,
            token_hash=hash_token(raw_refresh),
            expires_at=datetime.now(UTC) + timedelta(days=self.settings.refresh_token_expire_days),
        )
        access, expires_in = create_access_token(user_id, self.settings)
        return IssuedTokens(access_token=access, expires_in=expires_in, refresh_token=raw_refresh)

    async def _get_or_create_user(self, identity: GoogleIdentity) -> User:
        email = identity.email.strip().lower()

        user = await self.users.get_by_google_subject(identity.subject)
        if user is not None:
            user.name = identity.name
            user.avatar_url = identity.picture
            if user.email != email and await self.users.get_by_email(email) is None:
                user.email = email
            return user

        try:
            # Savepoint: a unique violation rolls back only this insert, not the whole session.
            async with self.db.begin_nested():
                return await self.users.create(
                    email=email,
                    name=identity.name,
                    google_subject_id=identity.subject,
                    avatar_url=identity.picture,
                )
        except IntegrityError:
            # Either a concurrent first login for the same Google account won the race...
            existing = await self.users.get_by_google_subject(identity.subject)
            if existing is not None:
                return existing
            # ...or the email belongs to a different Google account. Never auto-link.
            raise ConflictError("An account with this email already exists.") from None
