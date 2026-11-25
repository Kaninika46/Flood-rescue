"""Per-segment road status from Level 2 outputs (S3, prompt B6; used by src/routing/status.py).

Inputs for one evaluation folder (``src/level2_roads/evaluate.py``): the road submission CSV
(``Flooded`` per predicted segment), the post-event cloud masks and, with several flood-network
members, the disagreement rasters (share of members whose class differs from the ensemble's).

Status of each road segment, in this order:

* ``unknown``: at least ``cloud.min_fraction`` of the samples along it are under cloud/haze
* ``verify``: the mean member disagreement along it is at least ``ensemble.verify_disagreement``
* ``flooded`` / ``dry``: the ensemble's call

``confidence`` is 1 - mean disagreement (0 for unknown). Low-confidence segments are never hidden
(README, "Project rules"): they stay in the output with status ``verify``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATUSES = ("flooded", "dry", "unknown", "verify")


def sample_mask(raster: Path, line: Any, line_crs: str = "EPSG:4326", n: int = 50) -> np.ndarray:
    """Values of a raster at ``n`` evenly spaced points along a line (NaN outside)."""
    import geopandas as gpd
    import rasterio

    with rasterio.open(raster) as src:
        geom = gpd.GeoSeries([line], crs=line_crs).to_crs(src.crs).iloc[0]
        pts = [geom.interpolate(f, normalized=True) for f in np.linspace(0, 1, n)]
        vals = np.array([v[0] for v in src.sample([(p.x, p.y) for p in pts])], dtype=np.float32)
        inside = [
            (src.bounds.left <= p.x <= src.bounds.right)
            and (src.bounds.bottom <= p.y <= src.bounds.top)
            for p in pts
        ]
        vals[~np.array(inside)] = np.nan
        return vals


def decide(
    flooded: bool, cloud_share: float, disagreement: float, cfg: Mapping[str, Any]
) -> tuple[str, float]:
    """Status and confidence of one segment."""
    if cloud_share >= cfg.get("cloud", {}).get("min_fraction", 0.5):
        return "unknown", 0.0
    confidence = float(1.0 - disagreement) if disagreement == disagreement else 1.0
    if disagreement == disagreement and disagreement >= cfg.get("ensemble", {}).get(
        "verify_disagreement", 0.34
    ):
        return "verify", confidence
    return ("flooded" if flooded else "dry"), confidence


def segment_status(
    submission: pd.DataFrame, tile_paths: Mapping[str, Path], out_dir: Path, cfg: Mapping[str, Any]
) -> Any:
    """GeoDataFrame of road segments with ``status`` and ``confidence`` for every tile."""
    import geopandas as gpd
    import shapely.wkt

    from src.level2_roads.scoring import is_flooded, lon_lat_order, tile_bounds

    sub = submission.copy()
    sub.columns = [c.lower() for c in sub.columns]
    sub = sub[(sub["object"] == "Road") & sub["imageid"].isin(set(tile_paths))]
    rows = []
    for rec in sub.to_dict("records"):
        wkt = rec.get("wkt_geo")
        if not isinstance(wkt, str) or "EMPTY" in wkt:
            continue
        geom = lon_lat_order(shapely.wkt.loads(wkt), tile_bounds(tile_paths[rec["imageid"]]))
        stem = rec["imageid"]
        cloud_path = out_dir / "preds" / "cloud" / f"{stem}_cloud.tif"
        disagree_path = out_dir / "preds" / "flood_ensemble" / f"{stem}_flooddisagree.tif"
        cloud = sample_mask(cloud_path, geom) if cloud_path.exists() else np.array([0.0])
        disagree = (
            sample_mask(disagree_path, geom) / 255.0
            if disagree_path.exists()
            else np.array([np.nan])
        )
        cloud_share = float(np.nanmean(cloud > 0)) if np.isfinite(cloud).any() else 0.0
        mean_dis = float(np.nanmean(disagree)) if np.isfinite(disagree).any() else float("nan")
        status, conf = decide(is_flooded(rec["flooded"]), cloud_share, mean_dis, cfg)
        rows.append(
            {
                "image_id": stem,
                "status": status,
                "confidence": round(conf, 3),
                "model_flooded": is_flooded(rec["flooded"]),
                "cloud_share": round(cloud_share, 3),
                "disagreement": None if mean_dis != mean_dis else round(mean_dis, 3),
                "length_m": rec.get("length_m"),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
