"""Priority ranking of areas for rescue (prompt C3).

For each area (``exposure.exposure_by_area`` output):

* ``cut_off``: before the flood, a car could reach the area from a health facility (OSM
  amenity hospital or clinic) or a district centre; after removing the edges a car may not use
  (``src/routing/route.passable``) it cannot. Reachability uses the road graph without direction,
  from every facility at once.
* ``exposed_people`` from the exposure step.
* ``mean_confidence``: mean of (1 - 2 x ensemble std) of the flood map over the area; below
  ``verify_below`` the area is marked ``verify`` (shown, never hidden).

Order: cut-off areas first, then by exposed people (largest first). Output GeoJSON and CSV with
``priority`` 1..n.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from src.routing.graph import nearest_node
from src.routing.route import passable


def area_nodes(G: nx.MultiDiGraph, polygon: Any) -> list[Any]:
    """Graph nodes inside a lon/lat polygon, or the one nearest its centroid if none."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    shape = prep(polygon)
    inside = [n for n, d in G.nodes(data=True) if shape.contains(Point(d["x"], d["y"]))]
    if inside:
        return inside
    c = polygon.centroid
    return [nearest_node(G, c.x, c.y)]


def reachable_from(G: nx.MultiDiGraph, sources: Iterable[Any], keep_edge: Any = None) -> set[Any]:
    """Nodes reachable from any source over edges kept by ``keep_edge`` (undirected)."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        if keep_edge is None or keep_edge(data):
            H.add_edge(u, v)
    seen: set[Any] = set()
    for s in sources:
        if s in H and s not in seen:
            seen |= nx.node_connected_component(H, s)
    return seen


def mean_confidence(std_path: Path | None, polygon: Any) -> float:
    """Mean of 1 - 2 x std over the area (NaN without a raster or coverage)."""
    if std_path is None or not Path(std_path).exists():
        return float("nan")
    import rasterio
    from rasterio.mask import mask

    with rasterio.open(std_path) as src:
        import geopandas as gpd

        geom = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        try:
            arr, _ = mask(src, [geom], crop=True, filled=True, nodata=np.nan)
        except ValueError:  # no overlap
            return float("nan")
    values = arr[0][np.isfinite(arr[0])]
    return float(np.clip(1 - 2 * values, 0, 1).mean()) if values.size else float("nan")


def rank_areas(
    areas: Any,
    G: nx.MultiDiGraph,
    facilities: Iterable[tuple[float, float]],
    profiles: Mapping[str, Any],
    std_path: Path | None = None,
    vehicle: str = "car",
    verify_below: float = 0.6,
) -> Any:
    """Ranked GeoDataFrame: cut_off, exposed_people, mean_confidence, verify, priority."""
    sources = {nearest_node(G, lon, lat) for lon, lat in facilities}
    before = reachable_from(G, sources)
    after = reachable_from(G, sources, lambda d: passable(d, vehicle, profiles))
    out = areas.copy()
    cut, conf = [], []
    for geom in out.geometry:
        nodes = area_nodes(G, geom)
        was = any(n in before for n in nodes)
        cut.append(bool(was and not any(n in after for n in nodes)))
        conf.append(mean_confidence(std_path, geom))
    out["cut_off"] = cut
    out["mean_confidence"] = np.round(conf, 3)
    out["verify"] = [c == c and c < verify_below for c in conf]
    out = out.sort_values(["cut_off", "exposed_people"], ascending=[False, False])
    out["priority"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)
