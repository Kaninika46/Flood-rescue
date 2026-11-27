"""Backend settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """``DATABASE_URL`` (PostGIS), ``REDIS_URL`` (RQ) and ``JOB_MODE`` (``rq`` or ``sync``).

    The defaults hold no password: set both URLs from backend/.env (backend/README.md).
    """

    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://flood@localhost:5432/flood")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    job_mode: str = os.environ.get("JOB_MODE", "rq")
    cors_origins: tuple[str, ...] = tuple(
        os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","))


def get_settings() -> Settings:
    """Current settings (read at call time, so tests can change the environment)."""
    return Settings(
        database_url=os.environ.get("DATABASE_URL", Settings.database_url),
        redis_url=os.environ.get("REDIS_URL", Settings.redis_url),
        job_mode=os.environ.get("JOB_MODE", Settings.job_mode),
        cors_origins=tuple(os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")),
    )
