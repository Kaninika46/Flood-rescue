"""CAP 1.2 alerts, approval and the test-phone-only gateway (prompt C4).

Phone numbers here are from the 555-01xx range reserved for fiction, never real ones.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.alerts.approval import AlertRecord, ApprovalError, approve, cancel
from src.alerts.cap import NEPAL_TZ, AlertContent, build_alert, cap_polygon, cap_time, sms_text
from src.alerts.gateway import (
    ConsoleGateway,
    RecipientRefused,
    load_test_recipients,
    mask_number,
    send_alert,
)

SCHEMA = Path(__file__).parent / "data" / "CAP-v1.2.xsd"
TEST_PHONE = "+15555550100"
OTHER_PHONE = "+15555550199"
RING = [(86.70, 26.60), (86.80, 26.60), (86.80, 26.70), (86.70, 26.70)]


def _content() -> AlertContent:
    return AlertContent(
        river="Koshi",
        district="Saptari",
        shelter="the high school at Rajbiraj",
        depth_m=1.2,
        area_desc="Saptari district, Koshi floodplain",
        polygon=RING,
    )


def _record() -> AlertRecord:
    content = _content()
    xml = build_alert(
        content, sent=datetime(2024, 9, 27, 14, 5, tzinfo=NEPAL_TZ), identifier="test-1"
    )
    return AlertRecord(alert_id="test-1", cap_xml=xml, sms=sms_text(content))


def test_cap_validates_against_oasis_schema() -> None:
    from lxml import etree

    schema = etree.XMLSchema(etree.parse(str(SCHEMA)))
    doc = etree.fromstring(_record().cap_xml.encode("utf-8"))
    assert schema.validate(doc), schema.error_log
    ns = {"c": "urn:oasis:names:tc:emergency:cap:1.2"}
    assert doc.findtext("c:status", namespaces=ns) == "Exercise"
    assert doc.findtext("c:scope", namespaces=ns) == "Restricted"
    langs = [i.findtext("c:language", namespaces=ns) for i in doc.findall("c:info", ns)]
    assert langs == ["en-US", "ne-NP"]
    headline = doc.find("c:info", ns).findtext("c:headline", namespaces=ns)
    assert headline == (
        "Flood warning: Koshi rising in Saptari. Move to the high school at Rajbiraj now."
    )
    assert "not an official warning" in doc.find("c:info", ns).findtext(
        "c:senderName", namespaces=ns
    )


def test_cap_helpers() -> None:
    assert (
        cap_time(datetime(2024, 9, 27, 14, 5, 7, 123, tzinfo=NEPAL_TZ))
        == "2024-09-27T14:05:07+05:45"
    )
    assert cap_polygon(RING).startswith("26.60000,86.70000 ")
    assert cap_polygon(RING).endswith("26.60000,86.70000")
    with pytest.raises(ValueError):
        cap_time(datetime(2024, 9, 27))
    with pytest.raises(ValueError):
        build_alert(_content(), status="Actual")  # never real public alerts


def test_unapproved_alert_is_refused() -> None:
    record = _record()
    with pytest.raises(ApprovalError):
        send_alert(record, [TEST_PHONE], ConsoleGateway(), allowlist={TEST_PHONE})
    assert record.state == "draft"


def test_unknown_number_is_refused_and_nothing_is_sent() -> None:
    record = approve(_record(), "Officer A. Sharma")
    gateway = ConsoleGateway()
    with pytest.raises(RecipientRefused):
        send_alert(record, [TEST_PHONE, OTHER_PHONE], gateway, allowlist={TEST_PHONE})
    assert gateway.outbox == [] and record.state == "approved"


def test_missing_allowlist_refuses_everything(tmp_path: Path) -> None:
    record = approve(_record(), "Officer A. Sharma")
    assert load_test_recipients(tmp_path / "absent.yaml") == set()
    with pytest.raises(RecipientRefused):
        send_alert(record, [TEST_PHONE], ConsoleGateway(), allowlist=set())


def test_approved_alert_goes_to_test_phone_only(tmp_path: Path) -> None:
    allow = tmp_path / "test_recipients.yaml"
    allow.write_text(f'recipients:\n  - name: me\n    number: "{TEST_PHONE}"\n', encoding="utf-8")
    record = approve(_record(), "Officer A. Sharma")
    assert record.approved_by and record.approved_at
    gateway = ConsoleGateway()
    send_alert(record, [TEST_PHONE], gateway, allowlist=load_test_recipients(allow))
    assert gateway.outbox == [(TEST_PHONE, record.sms)]
    assert record.state == "sent" and record.sent_to == [TEST_PHONE]
    assert record.sms.startswith("EXERCISE")
    with pytest.raises(ApprovalError):  # never sent twice
        send_alert(record, [TEST_PHONE], gateway, allowlist={TEST_PHONE})


def test_approval_rules() -> None:
    with pytest.raises(ApprovalError):
        approve(_record(), "  ")
    record = approve(_record(), "Officer B")
    with pytest.raises(ApprovalError):
        approve(record, "Officer C")
    cancel(record)
    assert record.state == "cancelled"
    assert mask_number(TEST_PHONE) == "+155XXXXX100"
