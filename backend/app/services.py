"""Backend services: everything calls the code in src/ (no models are reimplemented here)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import MultiPolygon, Point, mapping
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Alert, FieldReport, FloodLayer, Priority, RoadStatus, Route, Scene
from src.alerts.approval import AlertRecord, approve, cancel, check_sendable
from src.alerts.cap import AlertContent, build_alert, polygon_from_geometry, sms_text
from src.alerts.gateway import ConsoleGateway, TwilioSmsGateway, load_test_recipients, send_alert
from src.common.config import data_path, load_config, repo_root
from src.routing.graph import edge_geometry, is_bridge
from src.routing.route import FIELD_REPORTS, load_vehicles, route

# --------------------------------------------------------------------------- graph


@lru_cache(maxsize=1)
def _base_graph() -> nx.MultiDiGraph:
    from src.routing.graph import load_graph

    cfg = load_config(repo_root() / "configs" / "routing.yaml")
    return load_graph(data_path(cfg["graph"]["path"]))


def get_graph() -> nx.MultiDiGraph:
    """FastAPI dependency: the road graph (tests override it with a toy graph)."""
    return _base_graph()


def edge_index(G: nx.MultiDiGraph) -> dict[str, tuple[Any, Any, Any]]:
    """edge_id -> (u, v, key)."""
    return {d["edge_id"]: (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)}


def latest_scene_id(session: Session) -> int | None:
    """Most recent processed scene (or None)."""
    return session.scalar(select(Scene.id).where(Scene.state == "done").order_by(Scene.id.desc()))


def overlay_statuses(session: Session, G: nx.MultiDiGraph, scene_id: int | None = None) -> None:
    """Copy statuses from road_status (scene + field reports) onto the graph (in place)."""
    scene_id = scene_id if scene_id is not None else latest_scene_id(session)
    index = edge_index(G)
    rows = session.scalars(select(RoadStatus).where(
        (RoadStatus.scene_id == scene_id) | (RoadStatus.source == "field")).order_by(RoadStatus.id))
    for row in rows:
        if row.edge_id in index:
            u, v, k = index[row.edge_id]
            G[u][v][k].update(status=row.status, confidence=row.confidence, source=row.source,
                              max_depth_m=row.max_depth_m if row.max_depth_m is not None
                              else float("nan"))
            if row.source == "field" and row.status == "dry":
                G[u][v][k]["confirmed_open"] = True


def compute_route(session: Session, G: nx.MultiDiGraph, vehicle: str,
                  start: tuple[float, float], end: tuple[float, float]) -> Route:
    """Route with the current statuses; stored and returned."""
    overlay_statuses(session, G)
    result = route(G, start, end, vehicle, load_vehicles())
    from shapely.geometry import shape

    line = shape(result["features"][0]["geometry"])
    row = Route(vehicle=vehicle, total_time_min=result["total_time_min"],
                n_verify=len(result["verify_segments"]), geojson=result,
                geom=from_shape(line, srid=4326))
    session.add(row)
    session.commit()
    return row


# --------------------------------------------------------------------------- field reports


REPORT_SNAP_M = 60.0  # a GPS-only report must be this close to a road


def add_field_report(session: Session, G: nx.MultiDiGraph, edge_id: str | None, report: str,
                     lon: float | None, lat: float | None, note: str | None,
                     reporter: str | None) -> FieldReport:
    """Store a report and record its status for both directions of the edge.

    Without ``edge_id`` (the rescuer app offline), the nearest road within ``REPORT_SNAP_M`` of
    the position is used; ``KeyError`` when there is none.
    """
    from src.routing.graph import nearest_edge

    if report not in FIELD_REPORTS:
        raise ValueError(f"report must be one of {sorted(FIELD_REPORTS)}")
    index = edge_index(G)
    if edge_id is None:
        hit = nearest_edge(G, lon, lat, REPORT_SNAP_M) if lon is not None and lat is not None \
            else None
        if hit is None:
            raise KeyError(f"no road within {REPORT_SNAP_M:.0f} m of the report position")
        edge_id = G[hit[0]][hit[1]][hit[2]]["edge_id"]
    if edge_id not in index:
        raise KeyError(edge_id)
    u, v, _ = index[edge_id]
    fr = FieldReport(edge_id=edge_id, report=report, note=note, reporter=reporter,
                     geom=from_shape(Point(lon, lat), srid=4326) if lon is not None else None)
    session.add(fr)
    status = FIELD_REPORTS[report]["status"]
    for a, b in ((u, v), (v, u)):
        for k, data in (G[a][b].items() if G.has_edge(a, b) else []):
            session.add(RoadStatus(edge_id=data["edge_id"], status=status, confidence=1.0,
                                   source="field", bridge=is_bridge(data),
                                   max_depth_m=None,
                                   geom=from_shape(edge_geometry(G, a, b, data), srid=4326)))
    session.commit()
    return fr


# --------------------------------------------------------------------------- alerts


def alert_from_priority(session: Session, priority_id: int, river: str, shelter: str) -> Alert:
    """CAP draft for one ranked area."""
    p = session.get(Priority, priority_id)
    if p is None:
        raise KeyError(priority_id)
    geom = to_shape(p.geom)
    content = AlertContent(river=river, district=p.area_name, shelter=shelter, depth_m=1.0,
                           area_desc=f"{p.area_name} (priority {p.priority})",
                           polygon=polygon_from_geometry(geom))
    xml = build_alert(content)
    ident = xml.split("<identifier>")[1].split("</identifier>")[0]
    alert = Alert(scene_id=p.scene_id, priority_id=p.id, identifier=ident, cap_xml=xml,
                  sms=sms_text(content))
    session.add(alert)
    session.commit()
    return alert


def _record(a: Alert) -> AlertRecord:
    return AlertRecord(alert_id=a.identifier, cap_xml=a.cap_xml, sms=a.sms, state=a.state,
                       approved_by=a.approved_by, approved_at=a.approved_at)


def approve_alert(session: Session, alert_id: int, officer: str) -> Alert:
    """draft -> approved, with the officer's name and time."""
    a = session.get(Alert, alert_id)
    if a is None:
        raise KeyError(alert_id)
    rec = approve(_record(a), officer)
    a.state, a.approved_by, a.approved_at = rec.state, rec.approved_by, rec.approved_at
    session.commit()
    return a


def cancel_alert(session: Session, alert_id: int, reason: str) -> Alert:
    """draft/approved -> cancelled, with the reason and time (the row is kept)."""
    a = session.get(Alert, alert_id)
    if a is None:
        raise KeyError(alert_id)
    rec = _record(a)
    rec.sent_at = a.sent_at
    cancel(rec, reason)
    a.state, a.cancel_reason, a.cancelled_at = rec.state, rec.cancel_reason, rec.cancelled_at
    session.commit()
    return a


def supersede_scene(session: Session, scene_id: int) -> Scene:
    """Mark a scene superseded (kept, but no longer the latest ``done`` scene)."""
    s = session.get(Scene, scene_id)
    if s is None:
        raise KeyError(scene_id)
    s.state = "superseded"
    session.commit()
    return s


def send_alert_row(session: Session, alert_id: int, recipients: list[str] | None,
                   gateway_name: str = "console") -> Alert:
    """Send to registered test phones only; refuses unapproved alerts and other numbers."""
    a = session.get(Alert, alert_id)
    if a is None:
        raise KeyError(alert_id)
    rec = _record(a)
    check_sendable(rec)
    numbers = recipients if recipients else sorted(load_test_recipients())
    gateway = TwilioSmsGateway() if gateway_name == "twilio" else ConsoleGateway()
    send_alert(rec, numbers, gateway)
    a.state, a.sent_at, a.sent_count = rec.state, rec.sent_at, len(rec.sent_to)
    session.commit()
    return a


# --------------------------------------------------------------------------- event results


def store_event(session: Session, scene: Scene, run: Any) -> None:
    """Save an ``src.event_pipeline.EventRun``: layers, road statuses, priorities, alert draft."""
    import geopandas as gpd

    for kind, key in (("probability", "prob"), ("confidence", "std"), ("depth", "depth")):
        if run.outputs.get(key):
            session.add(FloodLayer(scene_id=scene.id, kind=kind, path=str(run.outputs[key])))
    G = run.outputs.get("graph")
    if G is not None:
        rows = []
        for u, v, _k, data in G.edges(keys=True, data=True):
            depth = data.get("max_depth_m")
            rows.append(RoadStatus(
                scene_id=scene.id, edge_id=data["edge_id"], status=data.get("status", "dry"),
                confidence=data.get("confidence"), source=data.get("source", "level1"),
                max_depth_m=None if depth is None or depth != depth else float(depth),
                bridge=is_bridge(data), geom=from_shape(edge_geometry(G, u, v, data), srid=4326)))
        session.add_all(rows)
    if run.outputs.get("priorities"):
        ranked = gpd.read_file(run.outputs["priorities"])
        for r in ranked.itertuples():
            geom = r.geometry if r.geometry.geom_type == "MultiPolygon" else MultiPolygon([r.geometry])
            session.add(Priority(scene_id=scene.id, priority=int(r.priority), area_name=r.area_name,
                                 exposed_people=float(r.exposed_people),
                                 flooded_km2=float(r.flooded_km2), cut_off=bool(r.cut_off),
                                 verify=bool(r.verify), mean_confidence=_num(r.mean_confidence),
                                 geom=from_shape(geom, srid=4326)))
    rec = run.outputs.get("alert")
    if rec is not None:
        session.add(Alert(scene_id=scene.id, identifier=rec.alert_id, cap_xml=rec.cap_xml,
                          sms=rec.sms))
    scene.minutes = run.minutes
    scene.state = "done"
    session.commit()


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def lon_lat(geom: Any) -> dict[str, float | None]:
    """``{"lon", "lat"}`` of a point geometry (None when missing)."""
    if geom is None:
        return {"lon": None, "lat": None}
    p = to_shape(geom)
    return {"lon": p.x, "lat": p.y}


def verify_std() -> float:
    """Ensemble std above which a pixel or road is "verify" (configs/routing.yaml)."""
    cfg = load_config(repo_root() / "configs" / "routing.yaml")
    return float(cfg["status"]["max_std"])


def feature(geom: Any, props: dict[str, Any]) -> dict[str, Any]:
    """GeoJSON feature from a PostGIS geometry."""
    return {"type": "Feature", "geometry": mapping(to_shape(geom)) if geom is not None else None,
            "properties": props}


def scene_path_ok(path: str) -> bool:
    """A scene path must exist under $DATA_ROOT (the API never reads arbitrary files)."""
    from src.common.config import get_data_root

    p = Path(path).resolve()
    return p.is_file() and get_data_root() in p.parents
