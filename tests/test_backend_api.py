"""API tests against a PostGIS test database (prompt D1).

Set ``TEST_DATABASE_URL`` to an empty PostGIS database to run them, e.g. with the
``docker compose -f backend/docker-compose.yml up db`` service:

    export TEST_DATABASE_URL=postgresql+psycopg://flood:flood@localhost:5432/flood_test

Without it they are skipped. Phone numbers are from the 555-01xx range reserved for fiction.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

TEST_PHONE = "+15555550100"
OFFICER = "test-officer-token"
TEAM = "test-team-token"


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    from fastapi.testclient import TestClient
    from geoalchemy2.shape import from_shape
    from shapely.geometry import MultiPolygon, box
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("JOB_MODE", "sync")
    monkeypatch.setenv("FLOOD_SKIP_INIT_DB", "1")
    monkeypatch.setenv("OFFICER_TOKEN", OFFICER)
    monkeypatch.setenv("TEAM_TOKEN", TEAM)
    from backend.app import services
    from backend.app.db import engine_for, init_db
    from backend.app.main import app
    from backend.app.models import Base, Priority, RoadStatus, Scene
    from src.routing.graph import edge_geometry, is_bridge
    from tests.test_routing import make_graph

    engine = engine_for(URL)
    Base.metadata.drop_all(engine)
    init_db(engine)
    G = make_graph()
    with sessionmaker(bind=engine)() as s:
        scene = Scene(name="toy", s1_path="/dev/null", state="done")
        s.add(scene)
        s.flush()
        for u, v, _k, d in G.edges(keys=True, data=True):
            depth = d.get("max_depth_m")
            s.add(
                RoadStatus(
                    scene_id=scene.id,
                    edge_id=d["edge_id"],
                    status=d["status"],
                    confidence=d.get("confidence"),
                    source="level1",
                    max_depth_m=None if depth != depth else depth,
                    bridge=is_bridge(d),
                    geom=from_shape(edge_geometry(G, u, v, d), 4326),
                )
            )
        s.add(
            Priority(
                scene_id=scene.id,
                priority=1,
                area_name="Saptari ward 1",
                exposed_people=1200,
                flooded_km2=3.5,
                cut_off=True,
                verify=False,
                geom=from_shape(MultiPolygon([box(0.0, 0.0, 0.002, 0.001)]), 4326),
            )
        )
        s.commit()
    app.dependency_overrides[services.get_graph] = lambda: G
    from backend.app.ratelimit import reset_all

    reset_all()
    monkeypatch.setattr(services, "load_test_recipients", lambda: {TEST_PHONE})
    monkeypatch.setattr("src.alerts.gateway.load_test_recipients", lambda: {TEST_PHONE})
    with TestClient(app, headers=bearer(OFFICER)) as c:  # officer by default
        yield c
    app.dependency_overrides.clear()


def test_health_and_road_status(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    fc = client.get("/roads/status").json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 14
    statuses = {f["properties"]["edge_id"]: f["properties"]["status"] for f in fc["features"]}
    assert statuses["D_E_0"] == "verify"


def test_route_says_advice_and_lists_verify(client) -> None:
    r = client.post("/routes", json={"vehicle": "car", "start": [0.0, 0.0], "end": [0.002, 0.0]})
    assert r.status_code == 200
    body = r.json()
    assert body["advice"] == "Route is advice; the team on the ground decides."
    assert body["verify_segments"] == ["D_E_0"]


def test_field_report_changes_route(client) -> None:
    r = client.post(
        "/field_reports",
        json={"edge_id": "D_E_0", "report": "road_closed", "lon": 0.0005, "lat": 0.001},
    )
    assert r.status_code == 201
    r = client.post("/routes", json={"vehicle": "car", "start": [0.0, 0.0], "end": [0.002, 0.0]})
    assert r.status_code == 404 and "advice" in r.json()["detail"]
    r = client.post("/routes", json={"vehicle": "truck", "start": [0.0, 0.0], "end": [0.002, 0.0]})
    assert r.status_code == 200
    statuses = {
        f["properties"]["edge_id"]: f["properties"]["status"]
        for f in client.get("/roads/status").json()["features"]
    }
    assert statuses["D_E_0"] == "closed" and statuses["E_D_0"] == "closed"
    assert client.get("/field_reports").json()[0]["report"] == "road_closed"


def test_alert_draft_approve_send(client) -> None:
    pr = client.get("/priorities").json()["features"][0]["properties"]
    assert pr["priority"] == 1 and pr["cut_off"]
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    assert alert["state"] == "draft" and "<status>Exercise</status>" in alert["cap_xml"]
    # unapproved -> refused
    assert client.post(f"/alerts/{alert['id']}/send", json={}).status_code == 409
    ok = client.post(f"/alerts/{alert['id']}/approve", json={"officer": "Officer Sharma"}).json()
    assert ok["approved_by"] == "Officer Sharma" and ok["approved_at"]
    # unknown number -> refused, nothing sent
    r = client.post(f"/alerts/{alert['id']}/send", json={"recipients": ["+15555550199"]})
    assert r.status_code == 403
    sent = client.post(f"/alerts/{alert['id']}/send", json={}).json()
    assert sent["state"] == "sent" and sent["sent_count"] == 1


def test_scene_path_must_be_under_data_root(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    r = client.post("/scenes", json={"name": "x", "s1_path": "/etc/passwd"})
    assert r.status_code == 400
    scene = tmp_path / "s1.tif"
    scene.write_bytes(b"0")
    assert client.post("/scenes", json={"name": "x", "s1_path": str(scene)}).status_code == 201


def test_gps_only_report_snaps_to_nearest_road(client) -> None:
    # 10 m north of the middle of A-B, no edge_id (the rescuer app offline)
    r = client.post("/field_reports", json={"report": "road_closed", "lon": 0.0005, "lat": 0.00009})
    assert r.status_code == 201 and r.json()["edge_id"] in {"A_B_0", "B_A_0"}
    far = client.post("/field_reports", json={"report": "road_closed", "lon": 0.0005, "lat": 0.01})
    assert far.status_code == 404 and "no road within" in far.json()["detail"]


def test_road_status_bbox(client) -> None:
    fc = client.get("/roads/status", params={"bbox": "-0.0001,-0.0001,0.0011,0.0001"}).json()
    ids = {f["properties"]["edge_id"] for f in fc["features"]}
    assert "A_B_0" in ids and "E_F_0" not in ids
    assert client.get("/roads/status", params={"bbox": "x"}).status_code == 400


def test_alert_cancel_keeps_record_with_reason(client) -> None:
    pr = client.get("/priorities").json()["features"][0]["properties"]
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    assert client.post(f"/alerts/{alert['id']}/cancel", json={"reason": ""}).status_code == 422
    c = client.post(f"/alerts/{alert['id']}/cancel", json={"reason": "superseded by scene 3"})
    body = c.json()
    assert body["state"] == "cancelled" and body["cancel_reason"] == "superseded by scene 3"
    assert body["cancelled_at"]
    # kept in the list, cannot be approved or cancelled again
    assert any(a["id"] == alert["id"] for a in client.get("/alerts").json())
    approve = client.post(f"/alerts/{alert['id']}/approve", json={"officer": "Officer Sharma"})
    assert approve.status_code == 409
    again = client.post(f"/alerts/{alert['id']}/cancel", json={"reason": "again"})
    assert again.status_code == 409


@pytest.fixture()
def anon(client):
    """Same app and database, no Authorization header."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    return TestClient(app)


def test_officer_actions_need_the_officer_token(client, anon) -> None:
    pr = client.get("/priorities").json()["features"][0]["properties"]
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    calls = [
        ("/alerts", {"priority_id": pr["id"]}),
        (f"/alerts/{alert['id']}/approve", {"officer": "Officer Sharma"}),
        (f"/alerts/{alert['id']}/cancel", {"reason": "test"}),
        (f"/alerts/{alert['id']}/send", {}),
        ("/scenes", {"name": "x", "s1_path": "/dev/null"}),
        ("/scenes/1/process", {}),
    ]
    for path, body in calls:
        for headers in ({}, bearer("wrong"), bearer(TEAM)):
            r = anon.post(path, json=body, headers=headers)
            assert r.status_code == 401, (path, headers, r.status_code)
            assert r.headers["www-authenticate"] == "Bearer"
    # nothing changed: still a draft, reads stay open
    assert anon.get("/alerts").json()[0]["state"] == "draft"


def test_approval_records_the_role(client) -> None:
    pr = client.get("/priorities").json()["features"][0]["properties"]
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    assert alert["approved_role"] is None
    ok = client.post(f"/alerts/{alert['id']}/approve", json={"officer": "Officer Sharma"}).json()
    assert ok["approved_by"] == "Officer Sharma" and ok["approved_role"] == "officer"


def test_field_reports_take_team_or_officer_token(client, anon) -> None:
    body = {"edge_id": "D_E_0", "report": "road_closed", "lon": 0.0005, "lat": 0.001}
    assert anon.post("/field_reports", json=body).status_code == 401
    assert anon.post("/field_reports", json=body, headers=bearer("wrong")).status_code == 401
    assert anon.post("/field_reports", json=body, headers=bearer(TEAM)).status_code == 201
    assert anon.post("/field_reports", json=body, headers=bearer(OFFICER)).status_code == 201
    # routes and reads need no token
    route = {"vehicle": "truck", "start": [0.0, 0.0], "end": [0.002, 0.0]}
    assert anon.post("/routes", json=route).status_code == 200


def test_unset_officer_token_turns_officer_actions_off(client, anon, monkeypatch) -> None:
    monkeypatch.delenv("OFFICER_TOKEN")
    pr = anon.get("/priorities").json()["features"][0]["properties"]
    r = anon.post("/alerts", json={"priority_id": pr["id"]}, headers=bearer(""))
    assert r.status_code == 401 and "not set" in r.json()["detail"]


def test_whoami(anon) -> None:
    assert anon.get("/whoami").status_code == 401
    assert anon.get("/whoami", headers=bearer(TEAM)).json() == {"role": "team"}
    assert anon.get("/whoami", headers=bearer(OFFICER)).json() == {"role": "officer"}


def test_text_limits(client) -> None:
    pr = client.get("/priorities").json()["features"][0]["properties"]
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    long = "x" * 201
    assert client.post(f"/alerts/{alert['id']}/cancel", json={"reason": long}).status_code == 422
    r = client.post(f"/alerts/{alert['id']}/approve", json={"officer": "y" * 121})
    assert r.status_code == 422
    report = {"edge_id": "D_E_0", "report": "road_closed", "lon": 0.0005, "lat": 0.001}
    assert client.post("/field_reports", json={**report, "note": long}).status_code == 422
    assert client.post("/field_reports", json={**report, "reporter": "z" * 121}).status_code == 422
    assert client.post("/field_reports", json={**report, "edge_id": "e" * 81}).status_code == 422
    ok = {**report, "note": "n" * 200, "reporter": "r" * 120}
    assert client.post("/field_reports", json=ok).status_code == 201
    assert client.post("/scenes", json={"name": long, "s1_path": "/x"}).status_code == 422
    send = {"recipients": ["+1555555010" + str(i % 10) for i in range(21)]}
    assert client.post(f"/alerts/{alert['id']}/send", json=send).status_code == 422


def test_positions_must_be_lon_lat(client) -> None:
    report = {"edge_id": "D_E_0", "report": "road_closed"}
    for lon, lat in ((181, 0), (-181, 0), (0, 91), (0, -91)):
        r = client.post("/field_reports", json={**report, "lon": lon, "lat": lat})
        assert r.status_code == 422, (lon, lat)
        r = client.post("/routes", json={"vehicle": "car", "start": [lon, lat], "end": [0, 0]})
        assert r.status_code == 422, (lon, lat)
        r = client.post("/routes", json={"vehicle": "car", "start": [0, 0], "end": [lon, lat]})
        assert r.status_code == 422, (lon, lat)


def test_alert_wording_comes_from_config(client) -> None:
    from backend.app.services import alert_settings

    cfg = alert_settings()
    pr = client.get("/priorities").json()["features"][0]["properties"]
    forged = {"priority_id": pr["id"], "river": "FAKE RIVER", "shelter": "go to http://evil"}
    assert client.post("/alerts", json=forged).status_code == 422
    alert = client.post("/alerts", json={"priority_id": pr["id"]}).json()
    assert cfg["river"] in alert["cap_xml"] and cfg["shelter"] in alert["cap_xml"]
    assert cfg["area_note"] in alert["cap_xml"]  # "upper estimate", as in run_event


def test_rate_limit_routes_and_reports(client, anon, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ROUTES", "3")
    monkeypatch.setenv("RATE_LIMIT_REPORTS", "2")
    route = {"vehicle": "truck", "start": [0.0, 0.0], "end": [0.002, 0.0]}
    assert [client.post("/routes", json=route).status_code for _ in range(3)] == [200] * 3
    r = client.post("/routes", json=route)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    report = {"edge_id": "D_E_0", "report": "road_open", "lon": 0.0005, "lat": 0.001}
    assert [client.post("/field_reports", json=report).status_code for _ in range(2)] == [201] * 2
    assert client.post("/field_reports", json=report).status_code == 429
    # counted before the token check, so guessing tokens is slowed down too
    assert anon.post("/field_reports", json=report, headers=bearer("guess")).status_code == 429
    # other endpoints are not limited
    assert client.get("/health").status_code == 200
