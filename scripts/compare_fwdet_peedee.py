# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare our FwDET port with the original's PeeDee River result (prompt A9, optional check).

The PeeDee files in ``third_party/fwdet`` are Git LFS objects. Fetch them once:

    git -C third_party/fwdet lfs pull --include "test_case/PeeDee/*"
    python scripts/compare_fwdet_peedee.py

It rasterises ``WaterExtent_fixed.geojson`` onto ``NEDelevation.tif``, runs our port with the
FwDET defaults, and compares with ``Depth_Model_m.tif`` (the ArcGIS FwDET output): bias, MAE,
RMSE and correlation on cells both call flooded, plus a PNG in docs/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import rasterio
import typer
from rasterio.features import rasterize

from src.common.config import repo_root
from src.depth.fwdet import FwdetSettings, fwdet
from src.level1_radar.terrain import Grid, pixel_size_m

app = typer.Typer(add_completion=False, help=__doc__)
CASE = repo_root() / "third_party" / "fwdet" / "test_case" / "PeeDee"


@app.command()
def main(png: Path = typer.Option(None, help="Default: docs/figures/fwdet_peedee.png")) -> None:
    """Run the comparison and print the numbers."""
    dem_path = CASE / "NEDelevation.tif"
    if dem_path.stat().st_size < 1000:
        raise typer.Exit("PeeDee files are LFS pointers: run git lfs pull (see docstring)")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        grid = Grid(src.crs, src.transform, src.height, src.width)
    water = gpd.read_file(CASE / "WaterExtent_fixed.geojson").to_crs(grid.crs)
    flood = rasterize(
        ((g, 1) for g in water.geometry), out_shape=dem.shape, transform=grid.transform, fill=0
    ).astype(bool)
    dx, dy = pixel_size_m(grid)
    ours = fwdet(flood, dem, float(np.mean(dx)), float(dy), FwdetSettings()).depth
    with rasterio.open(CASE / "Depth_Model_m.tif") as ref_src:
        ref = np.full(dem.shape, np.nan, np.float32)
        rasterio.warp.reproject(
            rasterio.band(ref_src, 1),
            ref,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
        )
    both = np.isfinite(ours) & np.isfinite(ref) & (ref > 0)
    diff = ours[both] - ref[both]
    print(
        f"cells compared: {both.sum()}  (ours flooded {np.isfinite(ours).sum()}, "
        f"reference {np.isfinite(ref).sum()})"
    )
    print(
        f"bias {diff.mean():+.3f} m  MAE {np.abs(diff).mean():.3f} m  "
        f"RMSE {np.sqrt((diff**2).mean()):.3f} m  r {np.corrcoef(ours[both], ref[both])[0, 1]:.3f}"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes,
        [ref, ours, ours - ref],
        ["FwDET (ArcGIS) depth m", "our port depth m", "ours - reference m"],
        strict=True,
    ):
        im = ax.imshow(img, cmap="RdBu" if "-" in title else "Blues")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    out = png or repo_root() / "docs" / "figures" / "fwdet_peedee.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
