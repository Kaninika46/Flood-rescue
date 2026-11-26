"""Officer approval of alert drafts (prompt C4): draft -> approved -> sent.

An alert can be sent only when it is ``approved`` and carries both ``approved_by`` and
``approved_at`` (README, "Project rules"). Approval needs a named officer. A sent alert is
never sent again; a draft or approved alert can be cancelled with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

STATES = ("draft", "approved", "sent", "cancelled")


class ApprovalError(RuntimeError):
    """An action that the approval rules do not allow."""


@dataclass
class AlertRecord:
    """An alert and its approval trail."""

    alert_id: str
    cap_xml: str
    sms: str
    state: str = "draft"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    approved_by: str | None = None
    approved_at: datetime | None = None
    sent_at: datetime | None = None
    sent_to: list[str] = field(default_factory=list)
    cancel_reason: str | None = None
    cancelled_at: datetime | None = None


def approve(record: AlertRecord, officer: str, when: datetime | None = None) -> AlertRecord:
    """Mark a draft as approved by a named officer."""
    if record.state != "draft":
        raise ApprovalError(f"only drafts can be approved (state is {record.state})")
    if not officer or not officer.strip():
        raise ApprovalError("approval needs the officer's name")
    record.state = "approved"
    record.approved_by = officer.strip()
    record.approved_at = when or datetime.now(UTC)
    return record


def check_sendable(record: AlertRecord) -> None:
    """Raise unless the record is approved with ``approved_by`` and ``approved_at``."""
    if record.state != "approved":
        raise ApprovalError(f"alert {record.alert_id} is {record.state}, not approved")
    if not record.approved_by or record.approved_at is None:
        raise ApprovalError(f"alert {record.alert_id} lacks approved_by / approved_at")


def mark_sent(record: AlertRecord, recipients: list[str], when: datetime | None = None) -> None:
    """Record delivery (call only after a successful send)."""
    check_sendable(record)
    record.state = "sent"
    record.sent_at = when or datetime.now(UTC)
    record.sent_to = list(recipients)


def cancel(record: AlertRecord, reason: str, when: datetime | None = None) -> AlertRecord:
    """Cancel a draft or an approved alert that was not sent, keeping the reason and time.

    Cancelled alerts stay in the record (who drafted, approved and cancelled what, and why);
    they are never deleted.
    """
    if record.state == "sent":
        raise ApprovalError("a sent alert cannot be cancelled; issue a Cancel message instead")
    if record.state == "cancelled":
        raise ApprovalError(f"alert {record.alert_id} is already cancelled")
    if not reason or not reason.strip():
        raise ApprovalError("cancelling needs a reason")
    record.state = "cancelled"
    record.cancel_reason = reason.strip()
    record.cancelled_at = when or datetime.now(UTC)
    return record
