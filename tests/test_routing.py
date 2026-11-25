"""Routing rules on a hand-made 6-node graph (prompt B9).

    D ---verify--- E --------- F
    |              |(bridge)   |
    A ----dry----- B --0.2 m-- C

Edges are two-way. B-C has 0.2 m of water (car limit 0.15, truck limit 0.30); B-E is a bridge in
the flood zone; D-E is a "verify" segment. Nodes are ~111 m apart (0.001 degree).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.routing.graph import add_edge_ids
from src.routing.route import (
    ADVICE,
    NoRoute,
    apply_field_report,
    load_vehicles,
    route,
    unsafe_length_m,
)

NODES = {
    "A": (0.000, 0.000),
    "B": (0.001, 0.000),
    "C": (0.002, 0.000),
    "D": (0.000, 0.001),
    "E": (0.001, 0.001),
    "F": (0.002, 0.001),
}


def make_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph(crs="EPSG:4326")
    for n, (x, y) in NODES.items():
        G.add_node(n, x=x, y=y)
    spec = {
        ("A", "B"): {"status": "dry"},
        ("B", "C"): {"status": "flooded", "max_depth_m": 0.2},
        ("A", "D"): {"status": "dry"},
        ("D", "E"): {"status": "verify", "confidence": 0.4},
        ("E", "F"): {"status": "dry"},
        ("C", "F"): {"status": "dry"},
        ("B", "E"): {"status": "dry", "bridge": "yes", "in_flood_zone": True},
    }
    for (u, v), attrs in spec.items():
        for a, b in ((u, v), (v, u)):
            data = {
                "length": 111.0,
                "speed_kph": 36.0,
                "travel_time": 11.1,
                "max_depth_m": float("nan"),
                "confidence": 1.0,
                **attrs,
            }
            G.add_edge(a, b, key=0, **data)
    return add_edge_ids(G)


def nodes_of(result: dict) -> list[str]:
    ids = result["features"][0]["properties"]["edge_ids"]
    return [ids[0].split("_")[0]] + [i.split("_")[1] for i in ids]


@pytest.fixture()
def profiles() -> dict:
    return load_vehicles()


def test_car_avoids_deep_water_and_closed_bridge(profiles: dict) -> None:
    res = route(make_graph(), NODES["A"], NODES["C"], "car", profiles)
    assert nodes_of(res) == ["A", "D", "E", "F", "C"]
    assert res["advice"] == ADVICE and res["features"][0]["properties"]["advice"] == ADVICE
    assert res["verify_segments"] == ["D_E_0"]  # the verify edge is listed, not hidden


def test_truck_crosses_shallow_water(profiles: dict) -> None:
    res = route(make_graph(), NODES["A"], NODES["C"], "truck", profiles)
    assert nodes_of(res) == ["A", "B", "C"]


def test_verify_penalty_prefers_dry_detour(profiles: dict) -> None:
    G = make_graph()
    # car from A to E: via the bridge (closed) or D-E (verify, 3x) or A-B-C-F-E (deep water)
    res = route(G, NODES["A"], NODES["E"], "car", profiles)
    assert nodes_of(res) == ["A", "D", "E"]
    # open the bridge from the field: the dry 2-edge path through B now wins over verify
    apply_field_report(G, "B_E_0", "road_open")
    res = route(G, NODES["A"], NODES["E"], "car", profiles)
    assert nodes_of(res) == ["A", "B", "E"] and res["verify_segments"] == []


def test_field_report_closes_and_reroutes(profiles: dict) -> None:
    G = make_graph()
    apply_field_report(G, "D_E_0", "road_closed")
    with pytest.raises(NoRoute):
        route(G, NODES["A"], NODES["F"], "car", profiles)  # only deep water and a closed bridge
    res = route(G, NODES["A"], NODES["F"], "truck", profiles)
    assert nodes_of(res) == ["A", "B", "C", "F"]


def test_flooded_without_depth_is_closed(profiles: dict) -> None:
    G = make_graph()
    for _, _, d in G.edges(data=True):
        if d["edge_id"] in ("B_C_0", "C_B_0"):
            d["max_depth_m"] = float("nan")  # Level 2 says flooded, depth unknown
    res = route(G, NODES["A"], NODES["C"], "truck", profiles)
    assert "B_C_0" not in res["features"][0]["properties"]["edge_ids"]


def test_unsafe_length_is_zero_for_chosen_routes(profiles: dict) -> None:
    G = make_graph()
    for vehicle in ("car", "truck", "foot"):
        res = route(G, NODES["A"], NODES["C"], vehicle, profiles)
        ids = res["features"][0]["properties"]["edge_ids"]
        assert unsafe_length_m(G, ids, vehicle, profiles) == 0.0
    assert unsafe_length_m(G, ["B_C_0"], "car", profiles) == pytest.approx(111.0)


def test_boat_uses_only_deep_water(profiles: dict) -> None:
    G = make_graph()
    for _, _, d in G.edges(data=True):
        if d["edge_id"] in ("B_C_0", "C_B_0"):
            d["max_depth_m"] = 0.8
    res = route(G, NODES["B"], NODES["C"], "boat", profiles)
    assert nodes_of(res) == ["B", "C"]
    with pytest.raises(NoRoute):
        route(G, NODES["A"], NODES["C"], "boat", profiles)


def test_astar_matches_dijkstra(profiles: dict) -> None:
    G = make_graph()
    a = route(G, NODES["A"], NODES["C"], "car", profiles, astar=True)
    b = route(G, NODES["A"], NODES["C"], "car", profiles)
    assert nodes_of(a) == nodes_of(b)


def test_status_from_level1_rasters(tmp_path: Path) -> None:
    from affine import Affine

    from src.common.raster import write_raster
    from src.routing.status import StatusSettings, assign_status

    transform = Affine(0.0001, 0, -0.0005, 0, -0.0001, 0.0015)  # covers all six nodes
    depth = np.full((20, 30), np.nan, np.float32)
    depth[12:18, 13:23] = 0.4  # water around B-C (y ~ 0, x 0.0008-0.0018)
    std = np.full((20, 30), 0.05, np.float32)
    std[3:8, :] = 0.3  # low confidence along the D-E-F row
    for name, arr in (("depth", depth), ("std", std)):
        write_raster(
            tmp_path / f"{name}.tif", arr, crs="EPSG:4326", transform=transform, nodata=float("nan")
        )
    G = make_graph()
    for _, _, d in G.edges(data=True):
        d.pop("status")
        d["max_depth_m"] = float("nan")
    assign_status(
        G,
        depth=tmp_path / "depth.tif",
        std=tmp_path / "std.tif",
        settings=StatusSettings(sample_step_m=5.0),
    )
    by_id = {d["edge_id"]: d for _, _, d in G.edges(data=True)}
    assert by_id["B_C_0"]["status"] == "flooded" and by_id["B_C_0"]["max_depth_m"] == pytest.approx(
        0.4
    )
    assert by_id["A_D_0"]["status"] == "dry"
    assert by_id["E_F_0"]["status"] == "verify"
    assert by_id["B_E_0"]["in_flood_zone"]


def test_nearest_edge_for_gps_reports() -> None:
    from src.routing.graph import nearest_edge

    G = make_graph()
    # 10 m north of the middle of A-B (0.0005, 0.0)
    u, v, _k, d = nearest_edge(G, 0.0005, 0.00009)
    assert {u, v} == {"A", "B"} and 9.0 < d < 11.0
    assert nearest_edge(G, 0.0005, 0.01) is None  # ~1 km away: refused
