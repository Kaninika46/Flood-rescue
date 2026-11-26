"""Tests for src/level1_radar/terrain.py with synthetic rasters (no network)."""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_points

from src.common.raster import write_raster
from src.level1_radar import terrain
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


# --- local tiles, downloads and atomic writes (no network) ---------------------------------


def _small_grid() -> terrain.Grid:
    return terrain.Grid(CRS.from_epsg(4326), Affine(0.001, 0, 86.2, 0, -0.001, 26.8), 100, 100)


def test_local_tile_is_used_without_network(tmp_path: Path, monkeypatch) -> None:
    def no_network(url: str) -> bool:
        raise AssertionError(f"network used for {url}")

    monkeypatch.setattr(terrain, "remote_exists", no_network)
    url = terrain.tile_urls("hand", _small_grid())[0]
    local = terrain.local_tile_path(tmp_path, "hand", url)
    local.parent.mkdir(parents=True)
    local.write_bytes(b"tile")
    assert terrain.tile_sources("hand", _small_grid(), local_dir=tmp_path) == [str(local)]


def test_missing_local_tile_is_streamed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(terrain, "remote_exists", lambda url: True)
    sources = terrain.tile_sources("dem", _small_grid(), local_dir=tmp_path)
    assert sources and all(s.startswith("/vsicurl/https://") for s in sources)


class _FakeResponse:
    def __init__(self, body: bytes, length: int) -> None:
        self._body, self.headers = io.BytesIO(body), {"Content-Length": str(length)}

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeServer:
    """Answers HEAD and Range requests for ``body``; the listed request numbers come back short."""

    def __init__(self, body: bytes, short_requests: set[int] = frozenset()) -> None:
        self.body, self.short, self.ranges = body, set(short_requests), []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _FakeResponse:
        if request.get_method() == "HEAD":
            return _FakeResponse(b"", len(self.body))
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        piece = self.body[start : end + 1]
        if len(self.ranges) in self.short:
            piece = piece[: len(piece) // 2]  # connection dropped mid-piece
        return _FakeResponse(piece, len(piece))


def test_download_in_pieces_retries_a_short_piece(tmp_path: Path, monkeypatch) -> None:
    body = bytes(range(256)) * 40  # 10240 bytes
    server = _FakeServer(body, short_requests={2})
    monkeypatch.setattr(terrain.urllib.request, "urlopen", server)
    dest = tmp_path / "HAND" / "tile.tif"
    out = terrain.download_file("https://example/tile.tif", dest, delay_s=0, piece_bytes=4096)
    assert out == dest and dest.read_bytes() == body
    assert server.ranges == [(0, 4095), (4096, 8191), (4096, 8191), (8192, 10239)]
    assert not list(tmp_path.rglob("*.part"))


def test_download_resumes_from_part_file(tmp_path: Path, monkeypatch) -> None:
    body = bytes(range(256)) * 40
    dest = tmp_path / "tile.tif"
    (tmp_path / "tile.tif.part").write_bytes(body[:5000])  # left by an interrupted run
    server = _FakeServer(body)
    monkeypatch.setattr(terrain.urllib.request, "urlopen", server)
    terrain.download_file("https://example/tile.tif", dest, delay_s=0, piece_bytes=4096)
    assert dest.read_bytes() == body
    assert server.ranges[0] == (5000, 9095)


def test_download_gives_up_and_leaves_no_final_file(tmp_path: Path, monkeypatch) -> None:
    server = _FakeServer(b"x" * 100, short_requests={1, 2})
    monkeypatch.setattr(terrain.urllib.request, "urlopen", server)
    dest = tmp_path / "tile.tif"
    with pytest.raises(ConnectionError):
        terrain.download_file("https://example/tile.tif", dest, retries=2, delay_s=0)
    assert not dest.exists()


def test_terrain_rasters_written_atomically(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "scene.tif"
    grid = _small_grid()
    write_raster(
        scene,
        np.zeros((1, grid.height, grid.width), np.float32),
        crs=grid.crs,
        transform=grid.transform,
    )
    monkeypatch.setattr(
        terrain, "hand_for_grid", lambda g, c=None: np.ones((g.height, g.width), np.float32)
    )
    out = terrain.write_terrain_rasters(scene, tmp_path / "out", block=64, kinds=("hand",))
    assert out["hand"].exists() and not list((tmp_path / "out").glob("*.part"))
