"""docker-compose.yml keeps secrets in the git-ignored backend/.env (no Docker needed)."""

from __future__ import annotations

import re
import subprocess

import yaml

from src.common.config import repo_root

BACKEND = repo_root() / "backend"


def compose() -> dict:
    return yaml.safe_load((BACKEND / "docker-compose.yml").read_text())


def test_no_password_is_written_in_the_file() -> None:
    text = (BACKEND / "docker-compose.yml").read_text()
    svc = compose()["services"]
    assert svc["db"]["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
    env = svc["api"]["environment"]
    assert "${POSTGRES_PASSWORD:?" in env["DATABASE_URL"]
    assert env["REDIS_URL"].startswith("redis://:${REDIS_PASSWORD:?")
    assert "flood:flood@" not in text and "POSTGRES_PASSWORD: flood" not in text


def test_redis_requires_its_password() -> None:
    redis = compose()["services"]["redis"]
    assert "--requirepass" in " ".join(redis["command"])
    assert redis["environment"]["REDIS_PASSWORD"].startswith("${REDIS_PASSWORD:?")
    worker = " ".join(compose()["services"]["worker"]["command"])
    assert "$$REDIS_URL" in worker  # the worker connects with the password-bearing URL


def test_ports_are_published_on_localhost_only() -> None:
    for name, svc in compose()["services"].items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_env_file_is_ignored_and_the_example_holds_no_secrets() -> None:
    root = repo_root()
    ignored = subprocess.run(["git", "check-ignore", "-q", "backend/.env"], cwd=root)
    assert ignored.returncode == 0, "backend/.env must be git-ignored"
    example = subprocess.run(["git", "check-ignore", "-q", "backend/.env.example"], cwd=root)
    assert example.returncode == 1, "backend/.env.example must be committable"
    values = dict(
        line.split("=", 1) for line in (BACKEND / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert set(values) >= {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "OFFICER_TOKEN", "TEAM_TOKEN"}
    assert all(v == "change-me" for v in values.values())
    assert not re.search(r"\bflood:flood@", (BACKEND / "README.md").read_text())
