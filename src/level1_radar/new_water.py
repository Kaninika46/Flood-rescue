"""New water: flood = water in the post-flood scene and not in the pre-flood scene.

The U-Net finds *all* open water on one date, including the normal river channel and seasonal
water. Running the same model on a pre-flood scene of the same relative orbit
(``S1_<post date>_pre.tif``) and keeping only water that is new in the post scene gives a map
comparable to UNOSAT's "New Water / Water Increase" layers, and keeps the normal river channel
out of the exposure counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

NODATA = 255


def new_water(
    post_prob: np.ndarray, pre_prob: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """``(flood, no_data)``: water in post and not in pre; no data where either is NaN."""
    no_data = ~np.isfinite(post_prob) | ~np.isfinite(pre_prob)
    with np.errstate(invalid="ignore"):
        flood = (post_prob >= threshold) & ~(pre_prob >= threshold)
    return flood & ~no_data, no_data


def pre_scene_for(scene: Path) -> Path | None:
    """``<stem>_pre.tif`` next to a post scene, if it exists."""
    pre = scene.with_name(f"{scene.stem}_pre{scene.suffix}")
    return pre if pre.exists() else None


def write_new_water_mask(
    post_prob: Path, pre_prob: Path, out: Path, threshold: float = 0.5, block: int = 4096
) -> Path:
    """uint8 mask (1 new water, 0 not, 255 no data) on the post probability grid.

    Both probability rasters must be on the same grid (they are when both scenes come from one
    export, as in ``scripts/nepal_s1_download.py``).
    """
    from rasterio.windows import Window

    with rasterio.open(post_prob) as a, rasterio.open(pre_prob) as b:
        if (a.crs, a.transform, a.shape) != (b.crs, b.transform, b.shape):
            raise ValueError(f"{post_prob.name} and {pre_prob.name} are on different grids")
        profile = {
            "driver": "GTiff",
            "height": a.height,
            "width": a.width,
            "count": 1,
            "dtype": "uint8",
            "crs": a.crs,
            "transform": a.transform,
            "nodata": NODATA,
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(out, "w", **profile) as dst:
            for row in range(0, a.height, block):
                for col in range(0, a.width, block):
                    w = Window(col, row, min(block, a.width - col), min(block, a.height - row))
                    flood, no_data = new_water(
                        a.read(1, window=w).astype(np.float32),
                        b.read(1, window=w).astype(np.float32),
                        threshold,
                    )
                    mask = flood.astype(np.uint8)
                    mask[no_data] = NODATA
                    dst.write(mask[None], window=w)
    return out
