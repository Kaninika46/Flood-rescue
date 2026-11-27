"""Flood Rescue AI API (prompt D1).

    uvicorn backend.app.main:app --reload            # from the repo root, "flood" env

Every route answer says it is advice. Alerts are drafts until an officer approves them, and are
sent only to the registered test phones.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import networkx as nx
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app import services
from backend.app.auth import require_officer, require_team
from backend.app.db import get_session, init_db
from backend.app.ratelimit import rate_limited
from backend.app.models import Alert, FieldReport, FloodLayer, Priority, RoadStatus, Scene
from backend.app.settings import get_settings
from src.alerts.approval import ApprovalError
from src.alerts.gateway import RecipientRefused
from src.routing.route import ADVICE, NoRoute


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create PostGIS tables on start (skipped when tests manage the schema)."""
    if os.environ.get("FLOOD_SKIP_INIT_DB") != "1":
        init_db()
    yield


app = FastAPI(title="Flood Rescue AI", version="0.1.0", lifespan=lifespan,
              description="Student project. Routes are advice; alerts are exercises.")
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins),
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- schemas


# Input limits: free text is at most 200 characters (the database columns fit it), positions are
# real lon/lat, and the alert's river and shelter come from configs/priority.yaml, never from the
# request.
TEXT_MAX = 200  # notes, reasons, scene, officer and reporter names

Lon = Annotated[float, Field(ge=-180.0, le=180.0)]
Lat = Annotated[float, Field(ge=-90.0, le=90.0)]


class SceneIn(BaseModel):
    name: str = Field(min_length=1, max_length=TEXT_MAX)
    s1_path: str = Field(max_length=1024)


class RouteIn(BaseModel):
    vehicle: str = Field(pattern="^(foot|car|truck|boat)$")
    start: tuple[Lon, Lat]
    end: tuple[Lon, Lat]


class AlertIn(BaseModel):
    """Only the ranked area; any other field (e.g. free-text river or shelter) is refused."""

    model_config = ConfigDict(extra="forbid")
    priority_id: int


class ApproveIn(BaseModel):
    officer: str = Field(min_length=2, max_length=TEXT_MAX)


class CancelIn(BaseModel):
    reason: str = Field(min_length=3, max_length=TEXT_MAX)


class SendIn(BaseModel):
    recipients: list[Annotated[str, Field(max_length=20)]] | None = Field(default=None,
                                                                          max_length=20)
    gateway: str = Field(default="console", pattern="^(console|twilio)$")


class FieldReportIn(BaseModel):
    edge_id: str | None = Field(default=None, max_length=80)  # omitted: nearest road (GPS only)
    report: str = Field(pattern="^(road_open|road_closed|bridge_gone|flooded)$")
    lon: Lon | None = None
    lat: Lat | None = None
    note: str | None = Field(default=None, max_length=TEXT_MAX)
    reporter: str | None = Field(default=None, max_length=TEXT_MAX)


# --------------------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness."""
    return {"status": "ok"}


@app.get("/whoami")
def whoami(role: str = Depends(require_team)) -> dict[str, str]:
    """Role of the bearer token (the dashboard checks the officer token with it)."""
    return {"role": role}


@app.get("/scenes")
def list_scenes(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """All scenes, newest first."""
    rows = session.scalars(select(Scene).order_by(Scene.id.desc()))
    return [{"id": s.id, "name": s.name, "state": s.state, "minutes": s.minutes,
             "created_at": s.created_at} for s in rows]


@app.post("/scenes", status_code=201)
def create_scene(body: SceneIn, session: Session = Depends(get_session),
                 _role: str = Depends(require_officer)) -> dict[str, Any]:
    """Register a Sentinel-1 scene (the file must be under $DATA_ROOT)."""
    if not services.scene_path_ok(body.s1_path):
        raise HTTPException(400, "s1_path must be an existing file under $DATA_ROOT")
    scene = Scene(name=body.name, s1_path=body.s1_path)
    session.add(scene)
    session.commit()
    return {"id": scene.id, "state": scene.state}


@app.post("/scenes/{scene_id}/process", status_code=202)
def process_scene(scene_id: int, session: Session = Depends(get_session),
                  _role: str = Depends(require_officer)) -> dict[str, Any]:
    """Queue the event pipeline (prediction -> depth -> roads -> priority -> alert draft)."""
    from backend.app.jobs import enqueue_scene

    if session.get(Scene, scene_id) is None:
        raise HTTPException(404, "scene not found")
    return {"job_id": enqueue_scene(scene_id)}


@app.get("/layers")
def list_layers(scene_id: int | None = None,
                session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """Flood layers of a scene, or of all scenes: id, kind and PNG URL.

    Server file paths stay on the server (they reveal the disk layout and user name).
    """
    q = select(FloodLayer)
    if scene_id is not None:
        q = q.where(FloodLayer.scene_id == scene_id)
    return [{"id": r.id, "scene_id": r.scene_id, "kind": r.kind,
             "png": f"/layers/{r.id}/png"} for r in session.scalars(q.order_by(FloodLayer.id))]


@app.get("/layers/{layer_id}/png")
def layer_png(layer_id: int, session: Session = Depends(get_session)) -> Response:
    """Coloured PNG of a layer in Web Mercator; corners (lon/lat) in the X-Corners header."""
    import json

    from backend.app.render import corners, render_layer

    layer = session.get(FloodLayer, layer_id)
    if layer is None or not Path(layer.path).is_file():
        raise HTTPException(404, "layer not found")
    png, bounds = render_layer(Path(layer.path), layer.kind, services.verify_std())
    return Response(png, media_type="image/png", headers={
        "X-Corners": json.dumps(corners(bounds)), "Access-Control-Expose-Headers": "X-Corners",
        "Cache-Control": "max-age=60"})


@app.get("/layers/{layer_id}/corners")
def layer_corners(layer_id: int, session: Session = Depends(get_session)) -> list[list[float]]:
    """Lon/lat corners of a layer's PNG (top-left, top-right, bottom-right, bottom-left)."""
    from backend.app.render import corners, render_layer

    layer = session.get(FloodLayer, layer_id)
    if layer is None or not Path(layer.path).is_file():
        raise HTTPException(404, "layer not found")
    return corners(render_layer(Path(layer.path), layer.kind, services.verify_std())[1])


@app.get("/roads/status")
def road_status(scene_id: int | None = None, bbox: str | None = None,
                session: Session = Depends(get_session)) -> dict[str, Any]:
    """Road edges with status as GeoJSON (latest scene plus field reports).

    ``bbox=west,south,east,north`` (lon/lat) keeps only roads in that box (the app uses it).
    """
    scene_id = scene_id if scene_id is not None else services.latest_scene_id(session)
    q = select(RoadStatus).where(
        (RoadStatus.scene_id == scene_id) | (RoadStatus.source == "field"))
    if bbox:
        try:
            w, s, e, n = (float(x) for x in bbox.split(","))
        except ValueError as err:
            raise HTTPException(400, "bbox must be west,south,east,north") from err
        q = q.where(func.ST_Intersects(RoadStatus.geom, func.ST_MakeEnvelope(w, s, e, n, 4326)))
    rows = session.scalars(q.order_by(RoadStatus.id))
    latest: dict[str, RoadStatus] = {r.edge_id: r for r in rows}  # later rows win
    feats = [services.feature(r.geom, {"edge_id": r.edge_id, "status": r.status,
                                       "confidence": r.confidence, "max_depth_m": r.max_depth_m,
                                       "source": r.source, "bridge": r.bridge})
             for r in latest.values()]
    return {"type": "FeatureCollection", "features": feats}


@app.post("/routes", dependencies=[Depends(rate_limited("routes"))])
def request_route(body: RouteIn, session: Session = Depends(get_session),
                  G: nx.MultiDiGraph = Depends(services.get_graph)) -> dict[str, Any]:
    """Fastest safe route for a vehicle; always carries the advice text."""
    try:
        row = services.compute_route(session, G, body.vehicle, body.start, body.end)
    except NoRoute as err:
        raise HTTPException(404, f"{err}. {ADVICE}") from err
    return {"route_id": row.id, **row.geojson}


@app.get("/priorities")
def list_priorities(scene_id: int | None = None,
                    session: Session = Depends(get_session)) -> dict[str, Any]:
    """Ranked areas as GeoJSON (priority 1 first)."""
    scene_id = scene_id if scene_id is not None else services.latest_scene_id(session)
    rows = session.scalars(select(Priority).where(Priority.scene_id == scene_id)
                           .order_by(Priority.priority))
    feats = [services.feature(p.geom, {"id": p.id, "priority": p.priority,
                                       "area_name": p.area_name,
                                       "exposed_people": p.exposed_people,
                                       "flooded_km2": p.flooded_km2, "cut_off": p.cut_off,
                                       "verify": p.verify, "mean_confidence": p.mean_confidence})
             for p in rows]
    return {"type": "FeatureCollection", "features": feats}


def _alert_json(a: Alert) -> dict[str, Any]:
    return {"id": a.id, "identifier": a.identifier, "state": a.state, "sms": a.sms,
            "approved_by": a.approved_by, "approved_role": a.approved_role,
            "approved_at": a.approved_at, "sent_at": a.sent_at,
            "sent_count": a.sent_count, "priority_id": a.priority_id, "cap_xml": a.cap_xml,
            "cancel_reason": a.cancel_reason, "cancelled_at": a.cancelled_at}


@app.get("/alerts")
def list_alerts(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """All alerts, newest first."""
    return [_alert_json(a) for a in session.scalars(select(Alert).order_by(Alert.id.desc()))]


@app.post("/alerts", status_code=201)
def create_alert(body: AlertIn, session: Session = Depends(get_session),
                 _role: str = Depends(require_officer)) -> dict[str, Any]:
    """Draft a CAP alert for a ranked area."""
    try:
        return _alert_json(services.alert_from_priority(session, body.priority_id))
    except KeyError as err:
        raise HTTPException(404, "priority not found") from err


@app.post("/alerts/{alert_id}/approve")
def approve_alert(alert_id: int, body: ApproveIn, session: Session = Depends(get_session),
                  role: str = Depends(require_officer)) -> dict[str, Any]:
    """Officer approval (records approved_by, approved_role and approved_at)."""
    try:
        return _alert_json(services.approve_alert(session, alert_id, body.officer, role))
    except KeyError as err:
        raise HTTPException(404, "alert not found") from err
    except ApprovalError as err:
        raise HTTPException(409, str(err)) from err


@app.post("/alerts/{alert_id}/cancel")
def cancel_alert(alert_id: int, body: CancelIn, session: Session = Depends(get_session),
                 _role: str = Depends(require_officer)) -> dict[str, Any]:
    """Cancel a draft or approved alert with a reason; the record is kept, never deleted."""
    try:
        return _alert_json(services.cancel_alert(session, alert_id, body.reason))
    except KeyError as err:
        raise HTTPException(404, "alert not found") from err
    except ApprovalError as err:
        raise HTTPException(409, str(err)) from err


@app.post("/alerts/{alert_id}/send")
def send_alert(alert_id: int, body: SendIn, session: Session = Depends(get_session),
               _role: str = Depends(require_officer)) -> dict[str, Any]:
    """Send an approved alert to registered test phones only."""
    try:
        return _alert_json(services.send_alert_row(session, alert_id, body.recipients,
                                                   body.gateway))
    except KeyError as err:
        raise HTTPException(404, "alert not found") from err
    except ApprovalError as err:
        raise HTTPException(409, str(err)) from err
    except RecipientRefused as err:
        raise HTTPException(403, str(err)) from err


@app.get("/field_reports")
def list_field_reports(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """All field reports, newest first."""
    rows = session.scalars(select(FieldReport).order_by(FieldReport.id.desc()))
    return [{"id": r.id, "edge_id": r.edge_id, "report": r.report, "note": r.note,
             "reporter": r.reporter, "created_at": r.created_at,
             **services.lon_lat(r.geom)} for r in rows]


@app.get("/vehicles")
def vehicles() -> dict[str, Any]:
    """Vehicle profiles (config/vehicles.yaml) and the route advice text."""
    from src.routing.route import load_vehicles

    profiles = {k: v for k, v in load_vehicles().items() if k != "rules"}
    return {"vehicles": profiles, "advice": ADVICE}


@app.post("/field_reports", status_code=201, dependencies=[Depends(rate_limited("reports"))])
def post_field_report(body: FieldReportIn, session: Session = Depends(get_session),
                      G: nx.MultiDiGraph = Depends(services.get_graph),
                      _role: str = Depends(require_team)) -> dict[str, Any]:
    """Record a field report; the next route request uses it."""
    try:
        fr = services.add_field_report(session, G, body.edge_id, body.report, body.lon, body.lat,
                                       body.note, body.reporter)
    except KeyError as err:
        raise HTTPException(404, str(err).strip("'")) from err
    return {"id": fr.id, "edge_id": fr.edge_id, "report": fr.report}


@app.get("/offline/tiles.mbtiles")
def offline_tiles() -> Any:
    """Offline map for the rescuer app ($DATA_ROOT/tiles/koshi.mbtiles, scripts/make_mbtiles.py)."""
    from fastapi.responses import FileResponse

    from src.common.config import data_path

    path = data_path("tiles", "koshi.mbtiles")
    if not path.is_file():
        raise HTTPException(404, "no offline map yet: run scripts/make_mbtiles.py")
    return FileResponse(path, media_type="application/vnd.sqlite3", filename="koshi.mbtiles")
