"""CAP 1.2 alert drafts (prompt C4).

``build_alert`` returns an OASIS Common Alerting Protocol 1.2 XML document with one ``<info>``
block per language (English and Nepali templates in ``src/alerts/templates``). Safety defaults
(README, "Project rules"):

* ``status`` is ``Exercise`` and ``scope`` is ``Restricted`` with a restriction text saying it
  goes to the project's test phones only; the sender name says it is a student project and not
  an official warning. Nothing here impersonates an agency or addresses the public.
* An alert is a **draft** until an officer approves it (``src/alerts/approval.py``).

The XML validates against ``tests/data/CAP-v1.2.xsd`` (see ``tests/test_alerts.py``).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
TEMPLATES = Path(__file__).parent / "templates"
SENDER = "flood-rescue-ai@student-project.invalid"
SENDER_NAME = "Flood Rescue AI (student project exercise, not an official warning)"
RESTRICTION = "Exercise only: delivered to the project's registered test phones, never the public."
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))


@dataclass
class AlertContent:
    """What goes into the templates and the area block."""

    river: str
    district: str
    shelter: str
    depth_m: float
    area_desc: str
    polygon: Sequence[tuple[float, float]]  # (lon, lat) ring
    severity: str = "Severe"
    urgency: str = "Immediate"
    certainty: str = "Likely"
    hours_valid: int = 12
    extra: dict[str, Any] = field(default_factory=dict)


def load_template(language: str) -> dict[str, str]:
    """Template by short code (``en`` or ``ne``)."""
    return yaml.safe_load((TEMPLATES / f"{language}.yaml").read_text(encoding="utf-8"))


def cap_time(value: datetime) -> str:
    """CAP dateTime: seconds and an explicit offset, e.g. ``2024-09-27T14:05:00+05:45``."""
    if value.tzinfo is None:
        raise ValueError("CAP times need a time zone")
    text = value.replace(microsecond=0).isoformat()
    return text.replace("+00:00", "-00:00") if text.endswith("+00:00") else text


def cap_polygon(ring: Sequence[tuple[float, float]], max_points: int = 100) -> str:
    """CAP polygon text: ``lat,lon`` pairs, closed, at most ``max_points`` points."""
    pts = list(ring)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    if len(pts) < 4:
        raise ValueError("a CAP polygon needs at least 4 points (first = last)")
    if len(pts) > max_points:
        step = (len(pts) - 1) / (max_points - 1)
        pts = [pts[round(i * step)] for i in range(max_points - 1)] + [pts[0]]
    return " ".join(f"{lat:.5f},{lon:.5f}" for lon, lat in pts)


def polygon_from_geometry(geom: Any, tolerance_deg: float = 0.002) -> list[tuple[float, float]]:
    """Outer ring of the largest polygon of a lon/lat geometry, simplified."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return list(geom.simplify(tolerance_deg).exterior.coords)


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


def build_alert(
    content: AlertContent,
    languages: Sequence[str] = ("en", "ne"),
    sent: datetime | None = None,
    identifier: str | None = None,
    status: str = "Exercise",
) -> str:
    """CAP 1.2 XML for one alert with an ``<info>`` per language."""
    if status not in ("Exercise", "Test", "Draft"):
        raise ValueError("this project only produces Exercise, Test or Draft alerts")
    ET.register_namespace("", CAP_NS)
    sent = sent or datetime.now(NEPAL_TZ)
    root = ET.Element(f"{{{CAP_NS}}}alert")
    _sub(root, "identifier", identifier or f"flood-rescue-{uuid.uuid4()}")
    _sub(root, "sender", SENDER)
    _sub(root, "sent", cap_time(sent))
    _sub(root, "status", status)
    _sub(root, "msgType", "Alert")
    _sub(root, "scope", "Restricted")
    _sub(root, "restriction", RESTRICTION)
    values: Mapping[str, Any] = {
        "river": content.river,
        "district": content.district,
        "shelter": content.shelter,
        "depth_m": f"{content.depth_m:.1f}",
        **content.extra,
    }
    for lang in languages:
        tpl = load_template(lang)
        info = _sub(root, "info")
        _sub(info, "language", tpl["language"])
        _sub(info, "category", "Met")
        _sub(info, "event", tpl["event"])
        _sub(info, "responseType", "Evacuate")
        _sub(info, "urgency", content.urgency)
        _sub(info, "severity", content.severity)
        _sub(info, "certainty", content.certainty)
        _sub(info, "expires", cap_time(sent + timedelta(hours=content.hours_valid)))
        _sub(info, "senderName", SENDER_NAME)
        _sub(info, "headline", tpl["headline"].format(**values))
        _sub(info, "description", tpl["description"].format(**values))
        _sub(info, "instruction", tpl["instruction"].format(**values))
        area = _sub(info, "area")
        _sub(area, "areaDesc", content.area_desc)
        _sub(area, "polygon", cap_polygon(content.polygon))
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def sms_text(content: AlertContent, language: str = "en") -> str:
    """Short SMS text from the template (always starts with EXERCISE)."""
    tpl = load_template(language)
    return tpl["sms"].format(
        river=content.river,
        district=content.district,
        shelter=content.shelter,
        depth_m=f"{content.depth_m:.1f}",
    )
