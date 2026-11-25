"""Status and confidence of every road edge (prompt B9).

Edge attributes written by :func:`assign_status`:

* ``status``: ``flooded`` / ``dry`` / ``unknown`` / ``verify`` (``closed`` comes only from field
  reports, see ``route.apply_field_report``)
* ``confidence`` (0-1) and ``source`` (``level2``, ``level1``, ``none`` or ``field``)
* ``max_depth_m``: deepest water along the edge from the FwDET depth raster (NaN if unknown)
* ``in_flood_zone``: water (or a flooded neighbour edge) within ``flood_zone_m`` of the edge

Precedence: a Level 2 road segment covering the edge (``road_status.geojson``: optical, road
level) decides the status; otherwise the Level 1 rasters do: water deeper than 0 along the edge
means ``flooded``, and a mean ensemble std above ``max_std`` (low confidence) means ``verify``.
Edges outside every raster get ``outside_coverage`` (default ``verify``: no information is not
the same as dry). Depth is sampled in every case, so vehicle depth limits always apply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from src.routing.graph import edge_geometry


@dataclass(frozen=True)
class StatusSettings:
    """Thresholds for :func:`assign_status` (configs/routing.yaml)."""

    sample_step_m: float = 10.0
    max_std: float = 0.2
    level2_buffer_m: float = 10.0
    level2_min_overlap: float = 0.5
    flood_zone_m: float = 50.0
    outside_coverage: str = "verify"


class RasterSampler:
    """Samples one raster at points given in longitude / latitude."""

    def __init__(self, path: Path) -> None:
        import rasterio
        from pyproj import Transformer

        self.src = rasterio.open(path)
        self.to_src = Transformer.from_crs("EPSG:4326", self.src.crs, always_xy=True)

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Values (NaN for nodata or outside the raster)."""
        xs, ys = self.to_src.transform(lons, lats)
        b = self.src.bounds
        inside = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
        out = np.full(len(xs), np.nan, dtype=np.float64)
        if inside.any():
            vals = np.array(
                [
                    v[0]
                    for v in self.src.sample(
                        zip(np.asarray(xs)[inside], np.asarray(ys)[inside], strict=True)
                    )
                ],
                float,
            )
            if self.src.nodata is not None and not np.isnan(self.src.nodata):
                vals[vals == self.src.nodata] = np.nan
            out[inside] = vals
        return out


def sample_points(line: Any, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Points along a lon/lat line about every ``step_m`` metres (at least the two ends)."""
    from src.routing.graph import haversine_m

    coords = np.asarray(line.coords)
    seg = (
        haversine_m(coords[0, 0], coords[0, 1], coords[1:, 0], coords[1:, 1])
        if len(coords) > 1
        else np.array([0.0])
    )
    length = (
        float(
            sum(
                haversine_m(
                    coords[i, 0], coords[i, 1], coords[i + 1 : i + 2, 0], coords[i + 1 : i + 2, 1]
                )[0]
                for i in range(len(coords) - 1)
            )
        )
        if len(coords) > 1
        else float(seg.sum())
    )
    n = max(2, int(length / step_m) + 1)
    pts = [line.interpolate(f, normalized=True) for f in np.linspace(0, 1, n)]
    return np.array([p.x for p in pts]), np.array([p.y for p in pts])


def level2_lookup(road_status: Any, settings: StatusSettings) -> Any:
    """Level 2 segments buffered in metres, as a GeoDataFrame in a metric CRS."""
    gdf = road_status.copy()
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    gdf["zone"] = gdf.geometry.buffer(settings.level2_buffer_m)
    return gdf.set_geometry("zone"), utm


def assign_status(
    G: nx.MultiDiGraph,
    depth: Path | None = None,
    std: Path | None = None,
    level2: Any | None = None,
    settings: StatusSettings = StatusSettings(),
) -> nx.MultiDiGraph:
    """Write status, confidence, depth and flood-zone flags on every edge (in place)."""
    import geopandas as gpd

    depth_s = RasterSampler(depth) if depth else None
    std_s = RasterSampler(std) if std else None
    l2, l2_crs = (
        level2_lookup(level2, settings) if level2 is not None and len(level2) else (None, None)
    )
    for u, v, _k, data in G.edges(keys=True, data=True):
        if data.get("source") == "field":
            continue  # field reports win over models
        line = edge_geometry(G, u, v, data)
        lons, lats = sample_points(line, settings.sample_step_m)
        depths = depth_s.sample(lons, lats) if depth_s else np.full(len(lons), np.nan)
        stds = std_s.sample(lons, lats) if std_s else np.full(len(lons), np.nan)
        covered = bool(np.isfinite(depths).any() or np.isfinite(stds).any())
        if depth_s is not None:
            # inside the depth raster's footprint, NaN means "not flooded" (FwDET writes NaN
            # outside the flood); outside the footprint the depth stays unknown
            footprint = np.isfinite(depth_s.sample(lons, lats)) | np.isfinite(stds)
            water = np.where(footprint, np.nan_to_num(depths, nan=0.0), np.nan)
        else:
            water = depths
        data["max_depth_m"] = float(np.nanmax(water)) if np.isfinite(water).any() else float("nan")
        mean_std = float(np.nanmean(stds)) if np.isfinite(stds).any() else float("nan")
        status, conf, source = None, None, "none"
        if l2 is not None:
            geom = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(l2_crs).iloc[0]
            hits = l2[l2.intersects(geom)]
            if len(hits) and geom.length:
                best = max(hits.itertuples(), key=lambda h: geom.intersection(h.zone).length)
                if geom.intersection(best.zone).length >= settings.level2_min_overlap * geom.length:
                    status, conf, source = best.status, float(best.confidence), "level2"
        if status is None and covered:
            flooded = bool(np.nanmax(np.nan_to_num(water, nan=0.0)) > 0)
            conf = float(np.clip(1.0 - 2.0 * mean_std, 0.0, 1.0)) if mean_std == mean_std else 1.0
            status = "flooded" if flooded else "dry"
            if mean_std == mean_std and mean_std > settings.max_std:
                status = "verify"
            source = "level1"
        if status is None:
            status, conf = settings.outside_coverage, 0.0
        data.update(status=status, confidence=round(conf, 3), source=source)
    mark_flood_zone(G)
    return G


def mark_flood_zone(G: nx.MultiDiGraph) -> None:
    """``in_flood_zone``: the edge itself or an edge sharing a node is flooded / has water."""
    wet_nodes = set()
    for u, v, data in G.edges(data=True):
        depth = data.get("max_depth_m", float("nan"))
        if data.get("status") == "flooded" or (depth == depth and depth > 0):
            wet_nodes.update((u, v))
    for u, v, data in G.edges(data=True):
        depth = data.get("max_depth_m", float("nan"))
        data["in_flood_zone"] = bool(
            u in wet_nodes
            or v in wet_nodes
            or data.get("status") == "flooded"
            or (depth == depth and depth > 0)
        )


def status_counts(G: nx.MultiDiGraph) -> Mapping[str, int]:
    """Number of edges per status."""
    counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        counts[data.get("status", "none")] = counts.get(data.get("status", "none"), 0) + 1
    return counts
