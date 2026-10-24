from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

engine: AsyncEngine = create_async_engine(
    _settings.database_url,
    echo=_settings.db_echo,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_pre_ping=True,  # transparently replace connections the server dropped
    pool_recycle=1800,
    connect_args={"timeout": 10},  # asyncpg connect timeout, in seconds
)

# expire_on_commit=False: after commit, objects keep their loaded values. With the
# default (True), touching an attribute would trigger hidden I/O, which fails in async code.
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request.

    Services commit explicitly. If a request ends without a commit, the session
    closes and the transaction is rolled back.
    """
    async with AsyncSessionLocal() as session:
        yield session
