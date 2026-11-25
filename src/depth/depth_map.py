# SPDX-License-Identifier: GPL-2.0-or-later
# Command-line wrapper around src/depth/fwdet.py (a GPL-2.0 port of FwDET v2.1).
"""Water depth GeoTIFF from a flood map (prompt A9).

    # flood probability from src/level1_radar/predict.py, DEM fetched automatically
    python -m src.depth.depth_map --flood scene_flood_prob.tif --out scene_depth.tif

    # large scene: work on a 30 m UTM grid (the DEM's own resolution)
    python -m src.depth.depth_map --flood scene_flood_prob.tif --resolution-m 30 --out depth.tif

    # a mask instead of a probability (values 1 = water, 0 = dry, -1 = no data)
    python -m src.depth.depth_map --flood label.tif --mask --out depth.tif

Settings come from ``configs/depth.yaml`` (FwDET parameters); ``--set fwdet.min_slope_pct=0``
overrides them. Output: float32 depth in metres, NaN outside the flood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import typer
from rasterio.enums import Resampling

from src.common.config import load_config, repo_root
from src.common.logging import get_logger
from src.common.raster import write_raster
from src.depth.fwdet import FwdetSettings, fwdet
from src.level1_radar.terrain import Grid, dem_for_grid, pixel_size_m, utm_grid_for, warp_to_grid

log = get_logger("depth")
app = typer.Typer(add_completion=False, help=__doc__)


def flood_on_grid(
    path: Path, grid: Grid, is_mask: bool, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Flood mask and valid-data mask of ``path`` resampled onto ``grid``."""
    with rasterio.open(path) as src:
        same = (src.crs, src.transform, (src.height, src.width)) == (
            grid.crs,
            grid.transform,
            (grid.height, grid.width),
        )
        if same:
            values = src.read(1).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                values[values == src.nodata] = np.nan
        else:
            values = warp_to_grid(
                [str(path)], grid, Resampling.nearest if is_mask else Resampling.bilinear
            )
    if is_mask:
        valid = np.isfinite(values) & (values >= 0)
        return (values == 1) & valid, valid
    valid = np.isfinite(values)
    return (values >= threshold) & valid, valid


def depth_from_flood(
    flood_path: Path,
    settings: FwdetSettings,
    dem_path: Path | None = None,
    is_mask: bool = False,
    threshold: float = 0.5,
    resolution_m: float | None = None,
) -> tuple[np.ndarray, Grid]:
    """Depth array and its grid for a flood raster (DEM from file or fetched for the grid)."""
    grid = Grid.of(flood_path)
    if resolution_m:
        grid = utm_grid_for(grid, resolution_m)
    flood, valid = flood_on_grid(flood_path, grid, is_mask, threshold)
    dem = dem_for_grid(grid) if dem_path is None else warp_to_grid([str(dem_path)], grid)
    dx_rows, dy = pixel_size_m(grid)
    depth = np.full(flood.shape, np.nan, np.float32)
    rows, cols = np.nonzero(flood)
    if rows.size:  # run FwDET only on the flood's bounding box plus a margin
        pad = 64
        r0, r1 = max(0, rows.min() - pad), min(flood.shape[0], rows.max() + pad + 1)
        c0, c1 = max(0, cols.min() - pad), min(flood.shape[1], cols.max() + pad + 1)
        result = fwdet(
            flood[r0:r1, c0:c1], dem[r0:r1, c0:c1], float(np.mean(dx_rows)), float(dy), settings
        )
        depth[r0:r1, c0:c1] = result.depth
    depth[~valid] = np.nan
    return depth, grid


@app.command()
def main(
    flood: Path = typer.Option(..., help="Flood probability (0-1) or mask GeoTIFF."),
    out: Path = typer.Option(..., help="Output depth GeoTIFF."),
    dem: Path = typer.Option(None, help="DEM GeoTIFF (default: Copernicus GLO-30, fetched)."),
    mask: bool = typer.Option(False, "--mask", help="Input is a 1/0/-1 mask, not a probability."),
    threshold: float = typer.Option(0.5, help="Probability above which a pixel is flooded."),
    resolution_m: float = typer.Option(None, help="Work on a UTM grid of this pixel size."),
    config: Path = typer.Option(None, help="Default: configs/depth.yaml"),
    set_: list[str] = typer.Option([], "--set", help="Override, e.g. fwdet.min_slope_pct=0"),
) -> None:
    """Write the depth raster."""
    cfg = load_config(config or repo_root() / "configs" / "depth.yaml", set_)
    settings = FwdetSettings(**cfg["fwdet"])
    depth, grid = depth_from_flood(flood, settings, dem, mask, threshold, resolution_m)
    write_raster(
        out,
        depth,
        crs=grid.crs,
        transform=grid.transform,
        nodata=float("nan"),
        band_names=["water_depth_m"],
    )
    flooded = np.isfinite(depth)
    log.info(
        "wrote %s: %d flooded pixels, depth median %.2f m, p95 %.2f m",
        out,
        int(flooded.sum()),
        float(np.nanmedian(depth)) if flooded.any() else 0.0,
        float(np.nanpercentile(depth, 95)) if flooded.any() else 0.0,
    )


if __name__ == "__main__":
    app()
