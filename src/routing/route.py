"""Safe routes per vehicle on the flood-annotated road graph (prompt B9).

Rules, applied to every edge (status from ``src/routing/status.py``):

1. ``closed`` (field report) -> removed.
2. Bridges inside the flood zone are closed unless a field report confirmed them open
   (``confirmed_open``).
3. Water deeper than the vehicle's ``max_depth_m`` (``config/vehicles.yaml``) -> removed.
   ``flooded`` with unknown depth (Level 2 only sees "flooded") -> removed.
4. ``verify`` and ``unknown`` edges stay usable but cost ``verify_penalty`` (3x) time, and every
   one used is listed in the answer: low-confidence segments are never hidden.
5. Fastest path by travel time (Dijkstra; A* with a straight-line bound is optional).

Boats are handled separately: they may use only flooded edges at least ``min_depth_m`` deep.

Every answer carries the advice text "Route is advice; the team on the ground decides."
``apply_field_report`` changes an edge from the field and the caller re-routes.
``unsafe_length_m`` is the system metric: route length through water deeper than the vehicle's
limit (target 0).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from shapely.geometry import mapping

from src.common.config import repo_root
from src.routing.graph import edge_geometry, find_edge, haversine_m, is_bridge, nearest_node

ADVICE = "Route is advice; the team on the ground decides."
FIELD_REPORTS = {
    "road_open": {"status": "dry", "confirmed_open": True},
    "road_closed": {"status": "closed"},
    "bridge_gone": {"status": "closed"},
    "flooded": {"status": "flooded"},
}


class NoRoute(Exception):
    """No passable path between the two points for this vehicle."""


def load_vehicles(path: Path | None = None) -> dict[str, Any]:
    """``config/vehicles.yaml``."""
    path = path or repo_root() / "config" / "vehicles.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _depth(data: Mapping[str, Any]) -> float:
    value = data.get("max_depth_m", float("nan"))
    return float(value) if value is not None else float("nan")


def passable(data: Mapping[str, Any], vehicle: str, profiles: Mapping[str, Any]) -> bool:
    """Whether a vehicle may use an edge (rules 1-3, or the boat rule)."""
    status = data.get("status", "dry")
    depth = _depth(data)
    if status == "closed":
        return False
    if vehicle == "boat":
        return status == "flooded" and depth == depth and depth >= profiles["boat"]["min_depth_m"]
    rules = profiles.get("rules", {})
    if (
        rules.get("close_bridges_in_flood_zone", True)
        and is_bridge(dict(data))
        and data.get("in_flood_zone")
        and not data.get("confirmed_open")
    ):
        return False
    limit = profiles[vehicle]["max_depth_m"]
    if depth == depth and depth > limit:
        return False
    return not (status == "flooded" and depth != depth)


def travel_seconds(data: Mapping[str, Any], vehicle: str, profiles: Mapping[str, Any]) -> float:
    """Time to traverse an edge, before any penalty."""
    prof = profiles[vehicle]
    length = float(data.get("length", 0.0))
    if "speed_kph" in prof and (vehicle in ("foot", "boat") or "travel_time" not in data):
        return length / (prof["speed_kph"] / 3.6)
    base = data.get("travel_time")
    if base is None:
        base = length / (float(data.get("speed_kph", 30.0)) / 3.6)
    return float(base) / float(prof.get("speed_factor", 1.0))


def edge_cost(data: Mapping[str, Any], vehicle: str, profiles: Mapping[str, Any]) -> float:
    """Routing weight: travel time, x penalty for verify / unknown edges."""
    seconds = travel_seconds(data, vehicle, profiles)
    if data.get("status") in ("verify", "unknown"):
        seconds *= float(profiles.get("rules", {}).get("verify_penalty", 3.0))
    return seconds


def _best_key(G: nx.MultiDiGraph, u: Any, v: Any, vehicle: str, profiles: Mapping[str, Any]):
    options = [
        (edge_cost(d, vehicle, profiles), k)
        for k, d in G[u][v].items()
        if passable(d, vehicle, profiles)
    ]
    return min(options)[1]


def route(
    G: nx.MultiDiGraph,
    start: tuple[float, float],
    end: tuple[float, float],
    vehicle: str,
    profiles: Mapping[str, Any] | None = None,
    astar: bool = False,
) -> dict[str, Any]:
    """Fastest passable route from ``start`` to ``end`` (lon, lat) as a GeoJSON FeatureCollection.

    Raises:
        NoRoute: if the destination cannot be reached by this vehicle.
    """
    profiles = profiles or load_vehicles()
    if vehicle not in profiles or vehicle == "rules":
        raise ValueError(f"unknown vehicle {vehicle!r}")
    src, dst = nearest_node(G, *start), nearest_node(G, *end)

    def weight(u: Any, v: Any, edges: Mapping[Any, Mapping[str, Any]]) -> float | None:
        costs = [
            edge_cost(d, vehicle, profiles)
            for d in edges.values()
            if passable(d, vehicle, profiles)
        ]
        return min(costs) if costs else None  # None hides the edge from networkx

    try:
        if astar:
            fastest = (
                max(
                    30.0,
                    *(p.get("speed_kph", 0) for p in profiles.values() if isinstance(p, Mapping)),
                )
                / 3.6
            )

            def bound(a: Any, b: Any) -> float:
                na, nb = G.nodes[a], G.nodes[b]
                return float(haversine_m(na["x"], na["y"], [nb["x"]], [nb["y"]])[0]) / fastest

            nodes = nx.astar_path(G, src, dst, heuristic=bound, weight=weight)
        else:
            nodes = nx.dijkstra_path(G, src, dst, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as err:
        raise NoRoute(f"no passable route for {vehicle}") from err

    features, verify, total_s, total_m = [], [], 0.0, 0.0
    coords: list[tuple[float, float]] = []
    for u, v in itertools.pairwise(nodes):
        k = _best_key(G, u, v, vehicle, profiles)
        data = G[u][v][k]
        geom = edge_geometry(G, u, v, data)
        total_s += travel_seconds(data, vehicle, profiles)
        total_m += float(data.get("length", 0.0))
        part = list(geom.coords)
        coords += part if not coords else part[1:]
        if data.get("status") in ("verify", "unknown"):
            verify.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "edge_id": data.get("edge_id"),
                        "status": data["status"],
                        "confidence": data.get("confidence"),
                        "note": "verify on the ground",
                    },
                }
            )
    if len(coords) == 1:
        coords = coords * 2
    route_feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "kind": "route",
            "vehicle": vehicle,
            "total_time_min": round(total_s / 60, 2),
            "length_m": round(total_m, 1),
            "n_verify": len(verify),
            "advice": ADVICE,
            "edge_ids": [
                _edge_id(G, u, v, vehicle, profiles) for u, v in itertools.pairwise(nodes)
            ],
        },
    }
    features = [route_feature, *verify]
    return {
        "type": "FeatureCollection",
        "features": features,
        "advice": ADVICE,
        "verify_segments": [f["properties"]["edge_id"] for f in verify],
        "total_time_min": route_feature["properties"]["total_time_min"],
    }


def _edge_id(G: nx.MultiDiGraph, u: Any, v: Any, vehicle: str, profiles: Mapping[str, Any]) -> str:
    return G[u][v][_best_key(G, u, v, vehicle, profiles)].get("edge_id", f"{u}_{v}")


def apply_field_report(
    G: nx.MultiDiGraph, edge_id: str, report: str, both_directions: bool = True
) -> list[str]:
    """Apply a field report (road_open, road_closed, bridge_gone, flooded); returns edges changed.

    Field reports win over model statuses (``source = "field"``). The caller re-routes.
    """
    if report not in FIELD_REPORTS:
        raise ValueError(f"unknown report {report!r}; one of {sorted(FIELD_REPORTS)}")
    u, v, _k, data = find_edge(G, edge_id)
    targets = [data]
    if both_directions and G.has_edge(v, u):
        targets += list(G[v][u].values())
    changed = []
    for d in targets:
        d.update(FIELD_REPORTS[report], source="field", confidence=1.0)
        changed.append(d.get("edge_id"))
    return changed


def unsafe_length_m(
    G: nx.MultiDiGraph,
    edge_ids: list[str],
    vehicle: str,
    profiles: Mapping[str, Any] | None = None,
    depth_attr: str = "max_depth_m",
) -> float:
    """Length of the route through water deeper than the vehicle's limit (target 0)."""
    profiles = profiles or load_vehicles()
    limit = profiles[vehicle].get("max_depth_m", math.inf)
    total = 0.0
    for eid in edge_ids:
        _, _, _, data = find_edge(G, eid)
        depth = data.get(depth_attr, float("nan"))
        if depth == depth and depth > limit:
            total += float(data.get("length", 0.0))
    return total
