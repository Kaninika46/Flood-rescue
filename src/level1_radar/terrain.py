"""HAND and slope on exactly a target raster's grid (same CRS, transform and shape).

Sources (public, no account needed, streamed as Cloud-Optimised GeoTIFFs over HTTPS so only
the needed blocks are read):

* GLO-30 HAND, ``s3://glo-30-hand/v1/2021/`` (CC0), metres above nearest drainage.
* Copernicus GLO-30 DEM, ``s3://copernicus-dem-30m/``, for slope.

Both are 1x1 degree tiles at 1 arc-second (about 30 m) in EPSG:4326 named by their south-west
corner (``N26_00_E086_00`` covers 26-27 N, 86-87 E), with no nodata tag. Tiles that do not exist
(open ocean) are skipped; pixels they would cover become NaN.

HAND is resampled bilinearly. Slope is computed in degrees from the DEM after
bilinear resampling to the target grid plus a one-pixel border, so chip edges are correct.

Local tiles: a tile saved as ``$DATA_ROOT/terrain_tiles/{HAND,DEM}/<file name>.tif`` (see
``download_tiles`` and ``python scripts/prepare_hand.py tiles``) is always read from disk, with
no network request. Remote reads use HTTP timeouts and retries, so a stalled connection fails
after about a minute instead of waiting forever.
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

HAND_URL = "https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{name}_HAND.tif"
DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{name}_DEM/Copernicus_DSM_COG_10_{name}_DEM.tif"
)
EARTH_RADIUS_M = 6_371_008.8
# Timeouts so a stalled /vsicurl/ read fails (and is retried) instead of hanging forever.
GDAL_REMOTE_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "VSI_CACHE": "TRUE",
}
HTTP_TIMEOUT_S = 60.0
HTTP_RETRIES = 10
HTTP_RETRY_DELAY_S = 5.0
LOCAL_TILE_DIR = "terrain_tiles"  # under $DATA_ROOT
log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class Grid:
    """A target pixel grid."""

    crs: CRS
    transform: Affine
    height: int
    width: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in the grid's CRS."""
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return min(left, right), min(bottom, top), max(left, right), max(bottom, top)

    def buffered(self, pixels: int) -> Grid:
        """The same grid grown by ``pixels`` on every side."""
        shifted = self.transform @ Affine.translation(-pixels, -pixels)
        return Grid(self.crs, shifted, self.height + 2 * pixels, self.width + 2 * pixels)

    @classmethod
    def of(cls, path: str | os.PathLike[str]) -> Grid:
        """Grid of an existing raster file."""
        with rasterio.open(path) as src:
            return cls(src.crs, src.transform, src.height, src.width)


def tile_name(lat: int, lon: int) -> str:
    """Copernicus 1x1 degree tile name from its south-west corner, e.g. ``N26_00_E086_00``."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"


def tiles_for_bounds(bounds_wgs84: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """(lat, lon) south-west corners of all 1x1 degree tiles touching the bounds."""
    west, south, east, north = bounds_wgs84
    lats = range(math.floor(south), math.floor(north) + 1)
    lons = range(math.floor(west), math.floor(east) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def _with_retries(action: Callable[[], T], what: str, retries: int, delay_s: float) -> T:
    """Run ``action()``; on a network error wait and try again, up to ``retries`` times."""
    for attempt in range(1, retries + 1):
        try:
            return action()
        except urllib.error.HTTPError:
            raise  # a real HTTP answer (403/404/...) is not a network drop-out
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            if attempt == retries:
                raise
            log.warning(
                "%s failed (%s); retry %d/%d in %.0f s", what, err, attempt, retries, delay_s
            )
            time.sleep(delay_s)
    raise AssertionError("unreachable")


@lru_cache(maxsize=4096)
def remote_exists(url: str, timeout: float = 30.0) -> bool:
    """True if the URL answers a HEAD request with 200 (cached per process)."""

    def head() -> bool:
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout):
                return True
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):  # S3 answers 403 for a missing public key
                return False
            raise

    return _with_retries(head, f"HEAD {url}", HTTP_RETRIES, HTTP_RETRY_DELAY_S)


def download_file(
    url: str,
    dest: Path,
    timeout: float = HTTP_TIMEOUT_S,
    retries: int = HTTP_RETRIES,
    delay_s: float = HTTP_RETRY_DELAY_S,
    piece_bytes: int = 8 << 20,
) -> Path:
    """Download ``url`` to ``dest`` in HTTP Range pieces of ``piece_bytes``, resuming on drop-outs.

    Long single downloads stall on flaky connections, so the file is fetched piece by piece
    into ``<dest>.part``: each piece has its own timeout and retries, and a retry or a later
    run continues from the bytes already on disk. ``dest`` only appears once every byte has
    arrived, so a stopped download never leaves a truncated tile behind.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def total_size() -> int:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.headers["Content-Length"])

    total = _with_retries(total_size, f"HEAD {url}", retries, delay_s)
    if part.exists() and part.stat().st_size > total:
        part.unlink()  # our own leftover from a different file version: start again

    def fetch_piece() -> None:
        start = part.stat().st_size if part.exists() else 0
        end = min(start + piece_bytes, total) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if len(data) != end - start + 1:
            raise ConnectionError(f"short piece: {len(data)} of {end - start + 1} bytes")
        with part.open("ab") as out:
            out.write(data)

    while (part.stat().st_size if part.exists() else 0) < total:
        _with_retries(fetch_piece, f"download {url}", retries, delay_s)
    part.replace(dest)
    return dest


def default_tile_dir() -> Path | None:
    """``$DATA_ROOT/terrain_tiles`` if ``DATA_ROOT`` is set, else None."""
    from src.common.config import ConfigError, data_path

    try:
        return data_path(LOCAL_TILE_DIR)
    except ConfigError:
        return None


def local_tile_path(tile_dir: Path, kind: str, url: str) -> Path:
    """Where a downloaded tile lives: ``<tile_dir>/<KIND>/<file name>``."""
    return Path(tile_dir) / kind.upper() / url.rsplit("/", 1)[-1]


def tile_urls(kind: str, grid: Grid, margin_deg: float = 0.01) -> list[str]:
    """URLs of every 1x1 degree tile (existing or not) touching ``grid`` plus a margin."""
    template = {"hand": HAND_URL, "dem": DEM_URL}[kind]
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    padded = (west - margin_deg, south - margin_deg, east + margin_deg, north + margin_deg)
    return [template.format(name=tile_name(lat, lon)) for lat, lon in tiles_for_bounds(padded)]


def tile_sources(
    kind: str,
    grid: Grid,
    cache_dir: Path | None = None,
    margin_deg: float = 0.01,
    local_dir: Path | None = None,
) -> list[str]:
    """Paths GDAL can open for every existing tile under ``grid`` (``kind``: hand or dem).

    A tile already on disk in ``local_dir`` (default ``$DATA_ROOT/terrain_tiles``) or
    ``cache_dir`` is read from there without any network request. Otherwise, with
    ``cache_dir`` the whole tile is downloaded once and reused; without it the tile is
    streamed with ``/vsicurl/`` (only the needed blocks are fetched).
    """
    local_dirs = [d for d in (local_dir or default_tile_dir(), cache_dir) if d is not None]
    sources = []
    for url in tile_urls(kind, grid, margin_deg):
        local = next((p for d in local_dirs if (p := local_tile_path(d, kind, url)).exists()), None)
        if local is not None:
            sources.append(str(local))
            continue
        if not remote_exists(url):
            continue
        if cache_dir is None:
            sources.append("/vsicurl/" + url)
        else:
            sources.append(str(download_file(url, local_tile_path(cache_dir, kind, url))))
    return sources


def download_tiles(
    path: str | os.PathLike[str], tile_dir: Path, kinds: tuple[str, ...] = ("hand", "dem")
) -> dict[str, list[Path]]:
    """Download every existing HAND / DEM tile covering a raster's grid into ``tile_dir``.

    The DEM uses the grid plus a one-pixel border, as ``slope_for_grid`` does. Tiles already
    on disk are kept; missing tiles (open ocean) are skipped.
    """
    grid = Grid.of(path)
    out: dict[str, list[Path]] = {}
    for kind in kinds:
        target = grid.buffered(1) if kind == "dem" else grid
        out[kind] = []
        for url in tile_urls(kind, target):
            local = local_tile_path(tile_dir, kind, url)
            if not local.exists():
                if not remote_exists(url):
                    continue
                log.info("downloading %s", url)
                download_file(url, local)
            out[kind].append(local)
    return out


def warp_to_grid(
    sources: Iterable[str],
    grid: Grid,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: float | None = None,
) -> np.ndarray:
    """Mosaic ``sources`` onto ``grid`` (float32, NaN where no source covers a pixel)."""
    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    with rasterio.Env(**GDAL_REMOTE_OPTIONS):
        for path in sources:
            with rasterio.open(path) as src:
                part = np.full_like(out, np.nan)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=part,
                    src_nodata=src_nodata if src_nodata is not None else src.nodata,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
            take = np.isnan(out) & ~np.isnan(part)
            out[take] = part[take]
    return out


def pixel_size_m(grid: Grid) -> tuple[np.ndarray, float]:
    """Pixel width per row and pixel height in metres (handles geographic CRSs)."""
    res_x, res_y = abs(grid.transform.a), abs(grid.transform.e)
    if grid.crs.is_geographic:
        rows = np.arange(grid.height) + 0.5
        lats = grid.transform.f + rows * grid.transform.e
        meters_per_deg = math.pi / 180 * EARTH_RADIUS_M
        return res_x * meters_per_deg * np.cos(np.radians(lats)), res_y * meters_per_deg
    factor = grid.crs.linear_units_factor[1] if grid.crs.linear_units_factor else 1.0
    return np.full(grid.height, res_x * factor), res_y * factor


def slope_degrees(dem: np.ndarray, grid: Grid) -> np.ndarray:
    """Slope in degrees from a DEM on ``grid`` (central differences; NaN propagates)."""
    dx, dy = pixel_size_m(grid)
    dz_drow, dz_dcol = np.gradient(dem.astype(np.float64))
    gx = dz_dcol / dx[:, None]
    gy = dz_drow / dy
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def hand_for_grid(grid: Grid, cache_dir: Path | None = None) -> np.ndarray:
    """HAND in metres on ``grid`` (bilinear)."""
    return warp_to_grid(tile_sources("hand", grid, cache_dir), grid)


def slope_for_grid(grid: Grid, cache_dir: Path | None = None) -> np.ndarray:
    """Slope in degrees on ``grid``, computed with a one-pixel border to avoid edge effects."""
    big = grid.buffered(1)
    dem = warp_to_grid(tile_sources("dem", big, cache_dir), big)
    return slope_degrees(dem, big)[1:-1, 1:-1]


def iter_windows(height: int, width: int, block: int) -> Iterable[rasterio.windows.Window]:
    """Non-overlapping windows covering a ``height x width`` raster."""
    for row in range(0, height, block):
        for col in range(0, width, block):
            yield rasterio.windows.Window(
                col, row, min(block, width - col), min(block, height - row)
            )


def dem_for_grid(grid: Grid, cache_dir: Path | None = None) -> np.ndarray:
    """Copernicus GLO-30 elevation (m) on ``grid`` (bilinear)."""
    return warp_to_grid(tile_sources("dem", grid, cache_dir), grid)


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing a point (WGS 84)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def utm_grid_for(grid: Grid, resolution_m: float) -> Grid:
    """A north-up UTM grid at ``resolution_m`` covering ``grid``'s footprint."""
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    crs = CRS.from_epsg(utm_epsg((west + east) / 2, (south + north) / 2))
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, west, south, east, north)
    width = max(1, math.ceil((right - left) / resolution_m))
    height = max(1, math.ceil((top - bottom) / resolution_m))
    return Grid(crs, Affine(resolution_m, 0, left, 0, -resolution_m, top), height, width)


def write_terrain_rasters(
    path: str | os.PathLike[str],
    out_dir: Path,
    block: int = 2048,
    cache_dir: Path | None = None,
    kinds: tuple[str, ...] = ("hand", "slope"),
) -> dict[str, Path]:
    """Write ``<stem>_HAND.tif`` and ``<stem>_SLOPE.tif`` on any raster's grid, in blocks.

    Each file is written as ``<name>.part`` and renamed when complete, so an interrupted run
    never leaves a partial raster under the final name.
    """
    path = Path(path)
    grid = Grid.of(path)
    outputs = {}
    for kind in kinds:
        out = out_dir / f"{path.stem}_{kind.upper()}.tif"
        part = out.with_name(out.name + ".part")
        profile = {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": 1,
            "dtype": "float32",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": float("nan"),
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(part, "w", **profile) as dst:
            dst.set_band_description(1, "HAND_m" if kind == "hand" else "slope_deg")
            for window in iter_windows(grid.height, grid.width, block):
                sub = Grid(
                    grid.crs,
                    rasterio.windows.transform(window, grid.transform),
                    int(window.height),
                    int(window.width),
                )
                build = hand_for_grid if kind == "hand" else slope_for_grid
                dst.write(build(sub, cache_dir)[None], window=window)
        part.replace(out)
        outputs[kind] = out
    return outputs
