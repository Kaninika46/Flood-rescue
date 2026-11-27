"""A simple per-client-IP rate limit (sliding window, in memory, one process).

Limits per minute come from environment variables, read on every request:

* ``RATE_LIMIT_ROUTES`` (default 30) for ``POST /routes``
* ``RATE_LIMIT_REPORTS`` (default 60) for ``POST /field_reports``

Over the limit the API answers 429 with ``Retry-After``. The rescuer app keeps a report queued
after a 429 and sends it again later, so nothing is lost. Behind a Windows port-forward every
phone shares the Windows host's address, so the limit is then shared by all phones.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Callable
from threading import Lock

from fastapi import HTTPException, Request

WINDOW_S = 60.0
DEFAULTS = {"routes": ("RATE_LIMIT_ROUTES", 30), "reports": ("RATE_LIMIT_REPORTS", 60)}


class RateLimiter:
    """At most ``limit`` hits per key in any ``window_s`` seconds."""

    def __init__(self, window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic):
        self.window_s = window_s
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def hit(self, key: str, limit: int) -> float:
        """Record a hit; 0 when allowed, else the seconds until the next one is allowed."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= limit:
                return max(q[0] + self.window_s - now, 0.001)
            q.append(now)
            self._prune(now)
            return 0.0

    def _prune(self, now: float) -> None:
        """Forget clients with no hit in the window (keeps memory bounded)."""
        if len(self._hits) < 1000:
            return
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= now - self.window_s]:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiters: dict[str, RateLimiter] = {name: RateLimiter() for name in DEFAULTS}


def limit_for(name: str) -> int:
    """Requests per minute allowed for ``name`` (environment variable or default)."""
    env, default = DEFAULTS[name]
    try:
        return max(int(os.environ.get(env, default)), 1)
    except ValueError:
        return default


def rate_limited(name: str) -> Callable[[Request], None]:
    """FastAPI dependency: 429 when the client IP is over the ``name`` limit."""

    def check(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        wait = _limiters[name].hit(ip, limit_for(name))
        if wait:
            raise HTTPException(429, f"too many requests; try again in {math.ceil(wait)} s",
                                headers={"Retry-After": str(math.ceil(wait))})

    return check


def reset_all() -> None:
    """Clear every counter (tests)."""
    for limiter in _limiters.values():
        limiter.reset()
