"""Print tables, columns, indexes and constraints.  Run: python -m scripts.inspect_db"""

import asyncio

from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


def _describe(sync_conn: Connection) -> None:
    insp = inspect(sync_conn)
    for table in sorted(insp.get_table_names()):
        print(f"\n{table}")
        for col in insp.get_columns(table):
            print(f"  {col['name']:<20} {col['type']!s:<28} nullable={col['nullable']}")
        for idx in insp.get_indexes(table):
            print(f"  index  {idx['name']}: {idx['column_names']} unique={idx['unique']}")
        for uq in insp.get_unique_constraints(table):
            print(f"  unique {uq['name']}: {uq['column_names']}")
        for ck in insp.get_check_constraints(table):
            print(f"  check  {ck['name']}")


async def main() -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            await conn.run_sync(_describe)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
