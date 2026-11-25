"""Small geospatial helpers shared by the Nepal replay, exposure and routing code."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize

EARTH_RADIUS_M = 6_371_008.8


def pixel_area_km2(crs: CRS, transform: Affine, height: int) -> np.ndarray:
    """Area of one pixel in km^2 for each row (exact for geographic grids, constant otherwise)."""
    if crs.is_geographic:
        lat_top = transform.f + np.arange(height) * transform.e
        lat_bottom = lat_top + transform.e
        dlon = math.radians(abs(transform.a))
        band = np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom)))
        return (EARTH_RADIUS_M**2 * dlon * band) / 1e6
    factor = crs.linear_units_factor[1] if crs.linear_units_factor else 1.0
    return np.full(height, abs(transform.a * transform.e) * factor**2 / 1e6)


def mask_area_km2(mask: np.ndarray, crs: CRS, transform: Affine) -> float:
    """Total area of True pixels in km^2."""
    per_row = pixel_area_km2(crs, transform, mask.shape[0])
    return float((mask.sum(axis=1) * per_row).sum())


def rasterize_geometries(
    geometries: Iterable, shape: tuple[int, int], transform: Affine, all_touched: bool = False
) -> np.ndarray:
    """Boolean mask of polygons on a grid (geometries must already be in the grid's CRS)."""
    shapes = [(g, 1) for g in geometries if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, bool)
    return rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, all_touched=all_touched, dtype="uint8"
    ).astype(bool)


def read_vectors(folder: Path, patterns: tuple[str, ...] = ("*.shp", "*.geojson", "*.gpkg")):
    """All vector files in a folder as one GeoDataFrame in EPSG:4326 (column ``source``)."""
    import geopandas as gpd
    import pandas as pd

    frames = []
    for pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
            gdf["source"] = path.name
            frames.append(gdf)
    if not frames:
        raise FileNotFoundError(f"no vector files in {folder}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
