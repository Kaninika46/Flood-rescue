"""OpenStreetMap road graph for the safe-route engine (prompt B9).

    from src.routing.graph import download_graph, save_graph
    G = download_graph((86.8, 26.4, 87.3, 26.9), network_type="drive")   # west, south, east, north

Uses OSMnx (Overpass; run on the laptop). Bridges keep their ``bridge`` tag; every edge gets
``length`` (m), ``speed_kph`` and ``travel_time`` (s), and a stable ``edge_id`` "u_v_key".
Coordinates are WGS84 longitude / latitude (``x`` / ``y``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import LineString

BRIDGE_VALUES = {"yes", "true", "viaduct", "aqueduct", "boardwalk", "movable", "covered", "1"}


def download_graph(bbox: tuple[float, float, float, float], network_type: str = "drive") -> Any:
    """Road graph for a bounding box (west, south, east, north) with speeds and travel times."""
    import osmnx as ox

    ox.settings.useful_tags_way = list({*ox.settings.useful_tags_way, "bridge", "layer"})
    G = ox.graph_from_bbox(bbox, network_type=network_type, retain_all=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return add_edge_ids(G)


def save_graph(G: nx.MultiDiGraph, path: Path) -> Path:
    """GraphML file (OSMnx format)."""
    import osmnx as ox

    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return path


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Read a GraphML file written by :func:`save_graph`."""
    import osmnx as ox

    return add_edge_ids(ox.load_graphml(path))


def edge_id(u: Any, v: Any, key: Any) -> str:
    """Stable text id of a directed edge."""
    return f"{u}_{v}_{key}"


def add_edge_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Store ``edge_id`` on every edge (in place) and return the graph."""
    for u, v, k, data in G.edges(keys=True, data=True):
        data["edge_id"] = edge_id(u, v, k)
    return G


def iter_edges(G: nx.MultiDiGraph) -> Iterator[tuple[Any, Any, Any, dict[str, Any]]]:
    """``(u, v, key, data)`` for every edge."""
    yield from G.edges(keys=True, data=True)


def find_edge(G: nx.MultiDiGraph, eid: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Edge by ``edge_id``; raises ``KeyError`` if absent."""
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("edge_id") == eid:
            return u, v, k, data
    raise KeyError(eid)


def edge_geometry(G: nx.MultiDiGraph, u: Any, v: Any, data: dict[str, Any]) -> LineString:
    """The edge's geometry, or the straight line between its nodes."""
    geom = data.get("geometry")
    if geom is not None:
        return geom
    a, b = G.nodes[u], G.nodes[v]
    return LineString([(a["x"], a["y"]), (b["x"], b["y"])])


def is_bridge(data: dict[str, Any]) -> bool:
    """True when the OSM ``bridge`` tag marks a bridge (lists from merged ways included)."""
    value = data.get("bridge")
    values = value if isinstance(value, list) else [value]
    return any(str(v).lower() in BRIDGE_VALUES for v in values if v is not None)


def haversine_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres from one point to arrays of points."""
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - math.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_node(G: nx.MultiDiGraph, lon: float, lat: float) -> Any:
    """Node closest to a point (great-circle distance)."""
    nodes = list(G.nodes)
    xs = np.array([G.nodes[n]["x"] for n in nodes])
    ys = np.array([G.nodes[n]["y"] for n in nodes])
    return nodes[int(np.argmin(haversine_m(lon, lat, xs, ys)))]


def nearest_edge(
    G: nx.MultiDiGraph, lon: float, lat: float, max_m: float = 60.0
) -> tuple[Any, Any, Any, float] | None:
    """Edge closest to a point as ``(u, v, key, distance_m)``, or None beyond ``max_m``.

    Used for field reports sent with a GPS position only. Distances are point-to-segment in a
    local equirectangular frame (fine at road scale). The segment index is cached on the graph.
    """
    index = G.graph.get("_segment_index")
    if index is None:
        ends, owners = [], []
        for u, v, k, data in G.edges(keys=True, data=True):
            coords = list(edge_geometry(G, u, v, data).coords)
            for a, b in itertools.pairwise(coords):
                ends.append((*a, *b))
                owners.append((u, v, k))
        index = (np.asarray(ends, dtype=np.float64).reshape(-1, 4), owners)
        G.graph["_segment_index"] = index
    ends, owners = index
    if not len(owners):
        return None
    kx = 111_320.0 * math.cos(math.radians(lat))
    ky = 110_540.0
    ax, ay = (ends[:, 0] - lon) * kx, (ends[:, 1] - lat) * ky
    bx, by = (ends[:, 2] - lon) * kx, (ends[:, 3] - lat) * ky
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    t = np.clip(-(ax * dx + ay * dy) / np.where(seg2 > 0, seg2, 1.0), 0.0, 1.0)
    dist = np.hypot(ax + t * dx, ay + t * dy)
    i = int(np.argmin(dist))
    if dist[i] > max_m:
        return None
    u, v, k = owners[i]
    return u, v, k, float(dist[i])
