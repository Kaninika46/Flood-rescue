"""Tests for src/level1_radar/terrain.py with synthetic rasters (no network)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_points

from src.common.raster import write_raster
from src.level1_radar.terrain import (
    Grid,
    slope_degrees,
    tile_name,
    tiles_for_bounds,
    warp_to_grid,
)


def _plane(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return 100.0 + 2000.0 * (lon - 86.0) - 500.0 * (lat - 26.5)


def test_tile_names_and_bounds() -> None:
    assert tile_name(26, 86) == "N26_00_E086_00"
    assert tile_name(-16, -65) == "S16_00_W065_00"
    assert tiles_for_bounds((86.9, 26.5, 87.1, 26.6)) == [(26, 86), (26, 87)]
    assert tiles_for_bounds((-65.5, -16.2, -65.4, -15.9)) == [(-17, -66), (-16, -66)]


def test_resample_to_utm_chip_grid_is_exact_for_a_plane(tmp_path: Path) -> None:
    # source: geographic 1 arc-second tile carrying a linear field of lon/lat
    res = 1 / 3600
    src_transform = Affine(res, 0, 86.0, 0, -res, 27.0)
    rows, cols = np.mgrid[0:1800, 0:1800] + 0.5
    lon = 86.0 + cols * res
    lat = 27.0 - rows * res
    src = write_raster(
        tmp_path / "src.tif",
        _plane(lon, lat).astype(np.float32),
        crs="EPSG:4326",
        transform=src_transform,
    )

    # target: 64 x 48 pixels of 10 m in UTM 45N, placed at 86.25 E, 26.75 N (inside the source)
    (x0,), (y0,) = warp_points("EPSG:4326", "EPSG:32645", [86.25], [26.75])
    grid = Grid(CRS.from_epsg(32645), Affine(10.0, 0, round(x0), 0, -10.0, round(y0)), 48, 64)
    out = warp_to_grid([str(src)], grid)

    assert out.shape == (48, 64) and out.dtype == np.float32
    assert not np.isnan(out).any()
    r, c = np.mgrid[0:48, 0:64] + 0.5
    xs, ys = grid.transform @ (c.ravel(), r.ravel())
    lons, lats = warp_points(grid.crs, "EPSG:4326", xs, ys)
    expected = _plane(np.array(lons), np.array(lats)).reshape(48, 64)
    np.testing.assert_allclose(out, expected, atol=0.05)


def test_pixels_outside_sources_are_nan(tmp_path: Path) -> None:
    src = write_raster(
        tmp_path / "small.tif",
        np.ones((10, 10), np.float32),
        crs="EPSG:4326",
        transform=Affine(0.001, 0, 86.0, 0, -0.001, 27.0),
    )
    grid = Grid(CRS.from_epsg(4326), Affine(0.001, 0, 85.995, 0, -0.001, 27.0), 10, 10)
    out = warp_to_grid([str(src)], grid)
    assert np.isnan(out[:, :4]).all()
    assert (out[:, 6:] == 1).all()


def test_slope_of_tilted_plane_projected() -> None:
    grid = Grid(CRS.from_epsg(32645), Affine(10.0, 0, 0, 0, -10.0, 0), 20, 20)
    cols = np.arange(20) * 10.0
    dem = np.tile(cols * math.tan(math.radians(10)), (20, 1))
    slope = slope_degrees(dem, grid)
    np.testing.assert_allclose(slope, 10.0, atol=1e-6)


def test_slope_geographic_uses_metres() -> None:
    res = 1 / 3600  # about 30.9 m north-south
    grid = Grid(CRS.from_epsg(4326), Affine(res, 0, 86.0, 0, -res, 0.01), 20, 20)
    dy = res * math.pi / 180 * 6_371_008.8
    rows = np.arange(20)[:, None]
    dem = np.repeat(-rows * dy * math.tan(math.radians(5)), 20, axis=1)  # rises northwards
    slope = slope_degrees(dem, grid)
    np.testing.assert_allclose(slope, 5.0, atol=1e-3)


def test_buffered_grid() -> None:
    grid = Grid(CRS.from_epsg(4326), Affine(1.0, 0, 10.0, 0, -1.0, 20.0), 5, 4)
    big = grid.buffered(1)
    assert (big.height, big.width) == (7, 6)
    assert big.transform @ (1, 1) == grid.transform @ (0, 0)
