"""Bearer tokens for the actions that change things.

Two secrets come from environment variables and are never committed:

* ``OFFICER_TOKEN``: draft, approve, cancel and send alerts; register and process scenes.
* ``TEAM_TOKEN``: field reports from the rescuer app (the officer token works too).

Requests send ``Authorization: Bearer <token>``. A missing or wrong token gets 401. When a token
is not set on the server, the matching actions are refused for everyone (fail closed).
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False, description="OFFICER_TOKEN or TEAM_TOKEN")

ROLES = ("officer", "team")


def _matches(given: str, env_name: str) -> bool:
    expected = os.environ.get(env_name, "")
    return bool(expected) and secrets.compare_digest(given.encode(), expected.encode())


def role_for(token: str | None) -> str | None:
    """``"officer"``, ``"team"`` or None for a bearer token (constant-time comparison)."""
    if not token:
        return None
    if _matches(token, "OFFICER_TOKEN"):
        return "officer"
    if _matches(token, "TEAM_TOKEN"):
        return "team"
    return None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def _token(creds: HTTPAuthorizationCredentials | None) -> str | None:
    return creds.credentials if creds is not None and creds.scheme.lower() == "bearer" else None


def require_officer(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """FastAPI dependency: the officer token; returns the role."""
    if not os.environ.get("OFFICER_TOKEN"):
        raise _unauthorized("OFFICER_TOKEN is not set on the server; officer actions are off")
    if role_for(_token(creds)) != "officer":
        raise _unauthorized("officer token required")
    return "officer"


def require_team(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """FastAPI dependency: the team or the officer token; returns the role."""
    role = role_for(_token(creds))
    if role is None:
        raise _unauthorized("team token required")
    return role
