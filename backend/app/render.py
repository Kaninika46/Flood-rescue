"""Flood layers as PNG overlays for the dashboard (MapLibre ``image`` sources).

A GeoTIFF is warped to Web Mercator (the projection MapLibre draws image sources in), coloured
by layer kind and returned with the lon/lat of its four corners. Colours:

- ``probability``: blue, opacity rises with the flood probability
- ``depth``: light to dark blue over 0 to 3 m of water
- ``confidence``: diagonal orange hatching where the ensemble std is above the "verify" limit
  (``configs/routing.yaml`` ``status.max_std``); confident pixels stay clear
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np

MAX_SIDE = 2048
HATCH_PERIOD = 10  # pixels between hatch lines
DEPTH_MAX_M = 3.0


def _warp_to_mercator(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """First band in EPSG:3857 (NaN outside data) and its lon/lat bounds (W, S, E, N)."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    with rasterio.open(path) as src:
        dst_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        scale = max(width, height) / MAX_SIDE
        if scale > 1:
            width, height = int(width / scale), int(height / scale)
            transform = transform * transform.scale(scale, scale)
        out = np.full((height, width), np.nan, np.float32)
        reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=transform, dst_crs=dst_crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    bounds = transform_bounds(dst_crs, "EPSG:4326", left, bottom, right, top)
    return out, bounds


def colourise(values: np.ndarray, kind: str, max_std: float = 0.2) -> np.ndarray:
    """RGBA uint8 image of one layer."""
    h, w = values.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    valid = np.isfinite(values)
    v = np.nan_to_num(values)
    if kind == "probability":
        rgba[..., :3] = (33, 102, 172)
        alpha = np.clip((v - 0.1) / 0.9, 0, 1) * 210
        rgba[..., 3] = np.where(valid, alpha, 0).astype(np.uint8)
    elif kind == "depth":
        t = np.clip(v / DEPTH_MAX_M, 0, 1)[..., None]
        light, dark = np.array([198, 219, 239]), np.array([8, 48, 107])
        rgba[..., :3] = (light * (1 - t) + dark * t).astype(np.uint8)
        rgba[..., 3] = np.where(valid & (v > 0), 220, 0).astype(np.uint8)
    elif kind == "confidence":
        rows, cols = np.indices((h, w))
        stripe = ((rows + cols) % HATCH_PERIOD) < 3
        rgba[..., :3] = (230, 159, 0)
        rgba[..., 3] = np.where(valid & (v > max_std) & stripe, 235, 0).astype(np.uint8)
    else:
        raise ValueError(f"unknown layer kind {kind!r}")
    return rgba


@lru_cache(maxsize=16)
def _render_cached(path: str, mtime: float, kind: str, max_std: float) -> tuple[bytes, tuple]:
    from PIL import Image

    values, bounds = _warp_to_mercator(Path(path))
    buf = io.BytesIO()
    Image.fromarray(colourise(values, kind, max_std), "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue(), bounds


def render_layer(path: Path, kind: str, max_std: float = 0.2) -> tuple[bytes, tuple]:
    """PNG bytes and lon/lat bounds (W, S, E, N); cached until the file changes."""
    return _render_cached(str(path), path.stat().st_mtime, kind, max_std)


def corners(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    """MapLibre image-source corners: top-left, top-right, bottom-right, bottom-left."""
    w, s, e, n = bounds
    return [[w, n], [e, n], [e, s], [w, s]]
