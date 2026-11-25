"""Alert delivery to the project's own test phones only (prompt C4).

``send_alert`` refuses (raises, sends nothing) when

* the alert is not approved with ``approved_by`` and ``approved_at``, or
* any recipient is not listed in ``config/test_recipients.yaml`` (git-ignored; see
  ``config/test_recipients.example.yaml``), or that file is missing (fail closed).

Gateways are pluggable. ``ConsoleGateway`` (default) prints and keeps an outbox; it is what the
demo and the tests use. ``TwilioSmsGateway`` is one SMS adapter, configured from environment
variables (``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``, ``TWILIO_FROM``); it still goes through
the same allowlist. Nothing here can broadcast to the public.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from src.alerts.approval import AlertRecord, check_sendable, mark_sent
from src.common.config import repo_root
from src.common.logging import get_logger

log = get_logger("alerts")
E164 = re.compile(r"^\+[1-9]\d{6,14}$")


class RecipientRefused(PermissionError):
    """A recipient is not one of the registered test phones."""


class Gateway(Protocol):
    """Anything that can deliver one text to one phone number."""

    def send(self, number: str, text: str) -> str:
        """Deliver and return a provider message id."""


@dataclass
class ConsoleGateway:
    """Mock gateway: logs the message and keeps it in ``outbox``."""

    outbox: list[tuple[str, str]] = field(default_factory=list)

    def send(self, number: str, text: str) -> str:
        """Pretend to send."""
        self.outbox.append((number, text))
        log.info("[console gateway] to %s: %s", mask_number(number), text)
        return f"console-{len(self.outbox)}"


@dataclass
class TwilioSmsGateway:
    """SMS through Twilio's REST API (credentials from environment variables)."""

    account_sid: str = field(default_factory=lambda: os.environ.get("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = field(default_factory=lambda: os.environ.get("TWILIO_AUTH_TOKEN", ""))
    from_number: str = field(default_factory=lambda: os.environ.get("TWILIO_FROM", ""))

    def send(self, number: str, text: str) -> str:
        """Send one SMS; raises on any HTTP error."""
        import httpx

        if not (self.account_sid and self.auth_token and self.from_number):
            raise RuntimeError("Twilio credentials are not configured")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        resp = httpx.post(
            url,
            auth=(self.account_sid, self.auth_token),
            data={"To": number, "From": self.from_number, "Body": text},
            timeout=30,
        )
        resp.raise_for_status()
        return str(resp.json().get("sid"))


def mask_number(number: str) -> str:
    """``+97798XXXXX123`` style for logs (never log full numbers)."""
    return number[:4] + "X" * max(0, len(number) - 7) + number[-3:]


def load_test_recipients(path: Path | None = None) -> set[str]:
    """Numbers in ``config/test_recipients.yaml``; empty set if the file is missing."""
    path = path or repo_root() / "config" / "test_recipients.yaml"
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    numbers = {str(r["number"]).replace(" ", "") for r in data.get("recipients", [])}
    return {n for n in numbers if E164.match(n)}


def send_alert(
    record: AlertRecord,
    recipients: list[str],
    gateway: Gateway | None = None,
    allowlist: set[str] | None = None,
) -> list[str]:
    """Send an approved alert's SMS text to test phones; returns provider message ids."""
    check_sendable(record)
    allowed = load_test_recipients() if allowlist is None else allowlist
    if not allowed:
        raise RecipientRefused("no registered test phones (config/test_recipients.yaml)")
    cleaned = [r.replace(" ", "") for r in recipients]
    refused = [r for r in cleaned if r not in allowed]
    if refused or not cleaned:
        raise RecipientRefused(
            f"refused {len(refused)} recipient(s) not in the test-phone list: "
            f"{[mask_number(r) for r in refused]}"
            if refused
            else "no recipients given"
        )
    gateway = gateway or ConsoleGateway()
    ids = [gateway.send(number, record.sms) for number in cleaned]
    mark_sent(record, cleaned)
    return ids
