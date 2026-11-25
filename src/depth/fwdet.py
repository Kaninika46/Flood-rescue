# SPDX-License-Identifier: GPL-2.0-or-later
#
# Port of the Floodwater Depth Estimation Tool (FwDET) v2.1 to NumPy/SciPy.
# Source: https://github.com/csdms-contrib/fwdet (GPL-2.0),
# commit b47aec828bee313d373f37899e4d98d75937eb98,
# files FwDET_2p1_Standalone.py (ArcPy) and qgis_port/processing_scripts/fwdet_21.py (QGIS).
# Algorithm: Cohen et al. (2018, 2019), "Estimating floodwater depths from flood inundation maps
# and topography" / "The Floodwater Depth Estimation Tool (FwDET v2.0) ...".
# This file is distributed under the GPL-2.0 like its source (see THIRD_PARTY.md).
"""Water depth from a flood-extent raster and a DEM (FwDET v2.1).

Steps (same order as ``FwDET_2p1_Standalone.py``):

1. Clean the flood mask: fill holes smaller than ``min_hole_px`` and drop flood specks smaller
   than ``min_object_px`` (FwDET is sensitive to tiny holes, see the QGIS port's README).
2. Boundary cells: flooded cells with a non-flooded 8-neighbour. Their DEM elevation is the
   first guess of the water surface.
3. Smooth the boundary elevations ``smooth_iterations`` times with a 5x5 focal mean over
   boundary cells only (ArcPy ``FocalStatistics(Rectangle 5 5, MEAN, DATA)`` re-masked).
4. Drop "ocean" boundary cells where the 2-cell-radius minimum of the DEM is <= 0.
5. Slope filter as in FwDET v2.1: keep only boundary cells whose slope is **above**
   ``min_slope_pct`` (percent rise, default 0.5): on near-flat ground the extent edge says
   little about the water level. Optionally (prompt A9 of this project) also drop cells
   **steeper** than ``max_slope_deg``, e.g. a flood edge that runs up a valley wall.
6. Give every flooded cell the water-surface elevation of its nearest boundary cell
   (Euclidean, in metres: ``scipy.ndimage.distance_transform_edt(..., return_indices=True)``;
   replaces ArcPy CostAllocation / GRASS ``r.grow.distance`` with a neutral cost surface).
7. Depth = water surface - DEM, clipped at 0; NaN outside the flood or where the DEM is NaN.
8. Optional 3x3 low-pass filter of the depth inside the flood (FwDET's final step), repeated
   ``lowpass_passes`` times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FwdetSettings:
    """All FwDET parameters (defaults follow FwDET v2.1 where it has one)."""

    min_hole_px: int = 50
    min_object_px: int = 0
    smooth_iterations: int = 10
    smooth_size: int = 5
    remove_ocean: bool = True
    min_slope_pct: float = 0.5
    max_slope_deg: float | None = None
    lowpass_passes: int = 1


@dataclass
class FwdetResult:
    """Depth (m, NaN outside the flood), water surface elevation and the boundary used."""

    depth: np.ndarray
    water_surface: np.ndarray
    boundary: np.ndarray
    flood: np.ndarray


def clean_mask(flood: np.ndarray, min_hole_px: int, min_object_px: int) -> np.ndarray:
    """Fill small holes and drop small flood objects (8-connectivity)."""
    mask = flood.astype(bool)
    eight = np.ones((3, 3), bool)
    if min_hole_px > 0:
        holes, n = ndimage.label(~mask, structure=eight)
        if n:
            sizes = ndimage.sum(np.ones_like(holes), holes, index=np.arange(1, n + 1))
            border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
            small = np.flatnonzero(sizes < min_hole_px) + 1
            small = small[~np.isin(small, border)]  # background touching the edge is not a hole
            mask |= np.isin(holes, small)
    if min_object_px > 0:
        objects, n = ndimage.label(mask, structure=eight)
        if n:
            sizes = ndimage.sum(np.ones_like(objects), objects, index=np.arange(1, n + 1))
            mask &= np.isin(objects, np.flatnonzero(sizes >= min_object_px) + 1)
    return mask


def boundary_cells(flood: np.ndarray) -> np.ndarray:
    """Flooded cells with at least one non-flooded 8-neighbour inside the raster."""
    padded = np.pad(flood, 1, mode="edge")  # the raster edge itself is not a shoreline
    eroded = ndimage.binary_erosion(padded, structure=np.ones((3, 3), bool))[1:-1, 1:-1]
    return flood & ~eroded


def focal_mean_masked(values: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """``size x size`` mean of ``values`` over cells where ``mask`` is True (NaN elsewhere)."""
    v = np.where(mask & np.isfinite(values), values, 0.0)
    w = (mask & np.isfinite(values)).astype(np.float64)
    num = ndimage.uniform_filter(v, size, mode="constant")
    den = ndimage.uniform_filter(w, size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def slope_percent(dem: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Slope in percent rise from central differences."""
    gy, gx = np.gradient(dem.astype(np.float64), dy_m, dx_m)
    return 100.0 * np.hypot(gx, gy)


def fwdet(
    flood: np.ndarray,
    dem: np.ndarray,
    dx_m: float,
    dy_m: float,
    settings: FwdetSettings = FwdetSettings(),
) -> FwdetResult:
    """Run FwDET on a boolean flood mask and a DEM on the same grid.

    Args:
        flood: ``(H, W)`` True where flooded.
        dem: ``(H, W)`` elevation in metres (NaN = no data).
        dx_m, dy_m: Pixel width and height in metres (for slope and nearest-boundary distance).
        settings: Algorithm parameters.
    """
    if flood.shape != dem.shape:
        raise ValueError(f"flood {flood.shape} and DEM {dem.shape} differ")
    valid_dem = np.isfinite(dem)
    mask = clean_mask(flood, settings.min_hole_px, settings.min_object_px) & valid_dem
    boundary = boundary_cells(mask) & valid_dem
    elev = np.where(boundary, dem, np.nan).astype(np.float64)

    for _ in range(settings.smooth_iterations):
        smoothed = focal_mean_masked(elev, boundary, settings.smooth_size)
        elev = np.where(boundary, smoothed, np.nan)

    keep = boundary.copy()
    if settings.remove_ocean:
        disk = ndimage.generate_binary_structure(2, 1)
        disk = ndimage.iterate_structure(disk, 2)  # radius-2 "circle"
        dem_min = ndimage.minimum_filter(np.where(valid_dem, dem, np.inf), footprint=disk)
        keep &= dem_min > 0
    if settings.min_slope_pct > 0 or settings.max_slope_deg is not None:
        slope = slope_percent(np.where(valid_dem, dem, np.nan), dx_m, dy_m)
        if settings.min_slope_pct > 0:
            keep &= slope > settings.min_slope_pct
        if settings.max_slope_deg is not None:
            keep &= slope <= 100.0 * np.tan(np.radians(settings.max_slope_deg))
    keep &= np.isfinite(elev)

    surface = np.full(dem.shape, np.nan)
    depth = np.full(dem.shape, np.nan, dtype=np.float32)
    if keep.any() and mask.any():
        _, (rows, cols) = ndimage.distance_transform_edt(
            ~keep, sampling=(dy_m, dx_m), return_indices=True
        )
        nearest = elev[rows, cols]
        surface[mask] = nearest[mask]
        raw = np.clip(surface - dem, 0.0, None)
        depth[mask] = raw[mask]
        for _ in range(settings.lowpass_passes):
            depth = np.where(mask, focal_mean_masked(depth, mask, 3), np.nan).astype(np.float32)
    return FwdetResult(
        depth=depth, water_surface=surface.astype(np.float32), boundary=keep, flood=mask
    )
