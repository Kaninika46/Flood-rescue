"""Map (a): all water minus JRC permanent water, as a 0/1/255 mask on the probability grid.

The flood map used for exposure, priority and alerts (``configs/priority.yaml``, ``run_event``).
It also counts the normal river channel and seasonal water, so the numbers built on it are an
upper estimate; that is deliberate: for rescue, under-counting flooded people is worse than
over-counting (``docs/nepal_level1.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

NODATA = 255


def all_water(
    prob: np.ndarray, jrc: np.ndarray | None, threshold: float = 0.5, occ_limit: float = 50
) -> tuple[np.ndarray, np.ndarray]:
    """``(flood, no_data)``: probability >= threshold and not JRC occurrence >= ``occ_limit``.

    JRC NaN (outside its coverage) counts as not permanent.
    """
    no_data = ~np.isfinite(prob)
    with np.errstate(invalid="ignore"):
        flood = prob >= threshold
        if jrc is not None:
            flood &= ~(jrc >= occ_limit)
    return flood & ~no_data, no_data


def _jrc_window(jrc_src: rasterio.io.DatasetReader, dst: rasterio.io.DatasetReader, w: Window):
    out = np.full((int(w.height), int(w.width)), np.nan, np.float32)
    reproject(
        rasterio.band(jrc_src, 1),
        out,
        src_nodata=jrc_src.nodata,
        dst_transform=window_transform(w, dst.transform),
        dst_crs=dst.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return out


def write_all_water_mask(
    prob_path: Path,
    out: Path,
    threshold: float = 0.5,
    jrc_path: Path | None = None,
    occ_limit: float = 50,
    block: int = 4096,
) -> Path:
    """uint8 mask (1 flood, 0 dry, 255 no data) of map (a) on the probability grid."""
    jrc_src = rasterio.open(jrc_path) if jrc_path is not None and jrc_path.exists() else None
    try:
        with rasterio.open(prob_path) as src:
            profile = {
                "driver": "GTiff",
                "height": src.height,
                "width": src.width,
                "count": 1,
                "dtype": "uint8",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": NODATA,
                "compress": "deflate",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(out, "w", **profile) as dst:
                for row in range(0, src.height, block):
                    for col in range(0, src.width, block):
                        w = Window(
                            col, row, min(block, src.width - col), min(block, src.height - row)
                        )
                        jrc = _jrc_window(jrc_src, src, w) if jrc_src is not None else None
                        flood, no_data = all_water(
                            src.read(1, window=w).astype(np.float32), jrc, threshold, occ_limit
                        )
                        mask = flood.astype(np.uint8)
                        mask[no_data] = NODATA
                        dst.write(mask[None], window=w)
    finally:
        if jrc_src is not None:
            jrc_src.close()
    return out
