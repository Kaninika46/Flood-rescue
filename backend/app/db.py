"""Engine, sessions and table creation."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import Base
from backend.app.settings import get_settings


@lru_cache(maxsize=4)
def engine_for(url: str) -> Engine:
    """One engine per database URL."""
    return create_engine(url, pool_pre_ping=True)


def get_engine() -> Engine:
    """Engine for the configured ``DATABASE_URL``."""
    return engine_for(get_settings().database_url)


def init_db(engine: Engine | None = None) -> None:
    """Enable PostGIS and create the tables (idempotent)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:  # columns added after the first release
        conn.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS cancel_reason TEXT"))
        conn.execute(
            text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ")
        )
        conn.execute(
            text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS approved_role VARCHAR(20)")
        )
        # names may be up to 200 characters (API limit); widening a VARCHAR is instant
        conn.execute(text("ALTER TABLE alerts ALTER COLUMN approved_by TYPE VARCHAR(200)"))
        conn.execute(text("ALTER TABLE field_reports ALTER COLUMN reporter TYPE VARCHAR(200)"))


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    with factory() as session:
        yield session
