"""The API listens on 127.0.0.1 unless the phone demo is asked for (backend/app/__main__.py)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from backend.app.__main__ import app, choose_host, missing_tokens


@pytest.fixture()
def runs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture uvicorn.run calls instead of starting a server."""
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: calls.append({"target": target, **kw}))
    return calls


def test_choose_host() -> None:
    assert choose_host(False) == "127.0.0.1"
    assert choose_host(True) == "0.0.0.0"


def test_default_is_localhost(runs, monkeypatch) -> None:
    monkeypatch.delenv("OFFICER_TOKEN", raising=False)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert runs == [{"target": "backend.app.main:app", "host": "127.0.0.1", "port": 8000,
                     "reload": False}]


def test_phone_demo_needs_both_tokens(runs, monkeypatch) -> None:
    monkeypatch.setenv("OFFICER_TOKEN", "o")
    monkeypatch.delenv("TEAM_TOKEN", raising=False)
    assert missing_tokens() == ["TEAM_TOKEN"]
    result = CliRunner().invoke(app, ["--phone-demo"])
    assert result.exit_code == 2 and "TEAM_TOKEN" in result.output
    assert runs == []  # never started

    monkeypatch.setenv("TEAM_TOKEN", "t")
    result = CliRunner().invoke(app, ["--phone-demo"])
    assert result.exit_code == 0 and "phone's IP" in result.output
    assert runs[0]["host"] == "0.0.0.0"
