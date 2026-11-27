"""Bearer-token roles (backend/app/auth.py); no database needed."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.auth import require_officer, require_team, role_for


def creds(token: str, scheme: str = "Bearer") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


@pytest.fixture(autouse=True)
def tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFFICER_TOKEN", "o-secret")
    monkeypatch.setenv("TEAM_TOKEN", "t-secret")


def test_role_for() -> None:
    assert role_for("o-secret") == "officer"
    assert role_for("t-secret") == "team"
    assert role_for("wrong") is None
    assert role_for("") is None and role_for(None) is None


def test_officer_dependency() -> None:
    assert require_officer(creds("o-secret")) == "officer"
    for c in (None, creds("t-secret"), creds("wrong"), creds("o-secret", scheme="Basic")):
        with pytest.raises(HTTPException) as e:
            require_officer(c)
        assert e.value.status_code == 401


def test_team_dependency_accepts_officer_too() -> None:
    assert require_team(creds("t-secret")) == "team"
    assert require_team(creds("o-secret")) == "officer"
    with pytest.raises(HTTPException):
        require_team(None)


def test_unset_tokens_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OFFICER_TOKEN")
    monkeypatch.delenv("TEAM_TOKEN")
    assert role_for("") is None and role_for("o-secret") is None
    with pytest.raises(HTTPException) as e:
        require_officer(creds(""))
    assert "not set" in e.value.detail
