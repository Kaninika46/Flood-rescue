"""Flooded area and exposed people per administrative area (prompt C2).

The flood map (a probability GeoTIFF thresholded at ``threshold``, or a 1/0 mask with nodata 255
from ``scripts/nepal_level1.py``) is resampled onto the population grid (WorldPop, 100 m) as the
**flooded fraction** of each population cell (average resampling of the binary map). Then:

* ``method: majority`` (default, as in prompt C2): a population cell is flooded when at least half
  of it is flooded; exposed people = sum of the population of flooded cells
* ``method: fraction``: exposed people = sum of population x flooded fraction (no threshold)

Flooded km^2 = sum of flooded fraction x cell area, per area. Areas are assigned on the population
grid (cell centres), so every person is counted once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject

from src.common.geo import pixel_area_km2, rasterize_geometries


def flood_fraction_on_grid(
    flood_path: Path, threshold: float, dst_crs: Any, dst_transform: Any, shape: tuple[int, int]
) -> np.ndarray:
    """Flooded fraction (0-1, NaN where the flood map has no data) on a target grid."""
    with rasterio.open(flood_path) as src:
        is_mask = src.dtypes[0] in ("uint8", "int16", "int32")
        if is_mask:
            source, src_nodata = (
                rasterio.band(src, 1),
                src.nodata if src.nodata is not None else 255,
            )
            out = np.full(shape, np.nan, np.float32)
            reproject(
                source,
                out,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            return out
        prob = src.read(1).astype(np.float32)
        binary = np.where(np.isfinite(prob), (prob >= threshold).astype(np.float32), np.nan)
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": float("nan"),
        }
    with MemoryFile() as mem, mem.open(**profile) as tmp:
        tmp.write(binary[None])
        out = np.full(shape, np.nan, np.float32)
        reproject(
            rasterio.band(tmp, 1),
            out,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out


def exposure_by_area(
    flood_path: Path,
    population_path: Path,
    areas: Any,
    name_field: str,
    threshold: float = 0.5,
    method: str = "majority",
) -> Any:
    """GeoDataFrame of the areas with population, flooded_km2 and exposed_people."""
    if method not in ("majority", "fraction"):
        raise ValueError("method must be majority or fraction")
    with rasterio.open(population_path) as src:
        pop = src.read(1).astype(np.float64)
        if src.nodata is not None:
            pop[pop == src.nodata] = 0.0
        pop = np.nan_to_num(np.clip(pop, 0, None))
        crs, transform = src.crs, src.transform
    fraction = flood_fraction_on_grid(flood_path, threshold, crs, transform, pop.shape)
    covered = np.isfinite(fraction)
    frac = np.nan_to_num(fraction)
    cell_km2 = pixel_area_km2(crs, transform, pop.shape[0])[:, None]
    exposed_grid = np.where(frac >= 0.5, pop, 0.0) if method == "majority" else pop * frac
    zones = areas.to_crs(crs).reset_index(drop=True)
    out = zones.copy()
    pops, floods, exposed, analysed = [], [], [], []
    for geom in zones.geometry:
        inside = rasterize_geometries([geom], pop.shape, transform)
        pops.append(float(pop[inside].sum()))
        floods.append(float((frac * cell_km2)[inside].sum()))
        exposed.append(float(exposed_grid[inside].sum()))
        analysed.append(float((covered & inside).sum() / max(1, inside.sum())))
    out["area_name"] = zones[name_field].astype(str)
    out["population"] = np.round(pops, 1)
    out["flooded_km2"] = np.round(floods, 4)
    out["exposed_people"] = np.round(exposed, 1)
    out["analysed_share"] = np.round(analysed, 3)
    return out.to_crs("EPSG:4326")
