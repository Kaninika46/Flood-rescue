"""Per-client sliding-window rate limit (backend/app/ratelimit.py); no database needed."""

from __future__ import annotations

import pytest

from backend.app.ratelimit import RateLimiter, limit_for


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_limit_per_key_and_window() -> None:
    clock = Clock()
    rl = RateLimiter(window_s=60, clock=clock)
    assert [rl.hit("a", 3) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert rl.hit("a", 3) == pytest.approx(60.0)  # full window to wait
    assert rl.hit("b", 3) == 0.0  # another client is not affected
    clock.t += 30
    assert rl.hit("a", 3) == pytest.approx(30.0)
    clock.t += 30.001
    assert rl.hit("a", 3) == 0.0  # the oldest hits left the window


def test_refused_hits_do_not_extend_the_wait() -> None:
    clock = Clock()
    rl = RateLimiter(window_s=60, clock=clock)
    rl.hit("a", 1)
    for _ in range(50):
        clock.t += 1
        rl.hit("a", 1)
    clock.t += 10
    assert rl.hit("a", 1) == 0.0


def test_idle_clients_are_forgotten() -> None:
    clock = Clock()
    rl = RateLimiter(window_s=60, clock=clock)
    for i in range(1000):
        rl.hit(f"ip{i}", 5)
    clock.t += 61
    rl.hit("new", 5)
    assert len(rl._hits) == 1


def test_limits_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATE_LIMIT_ROUTES", raising=False)
    assert limit_for("routes") == 30 and limit_for("reports") == 60
    monkeypatch.setenv("RATE_LIMIT_ROUTES", "5")
    assert limit_for("routes") == 5
    monkeypatch.setenv("RATE_LIMIT_ROUTES", "lots")
    assert limit_for("routes") == 30
    monkeypatch.setenv("RATE_LIMIT_ROUTES", "0")
    assert limit_for("routes") == 1
