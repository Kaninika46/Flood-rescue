"""SpaceNet 8 split files and dataset.

Adapted from ``third_party/spacenet8/baseline/datasets/datasets.py`` (Apache-2.0, commit 30ca30c):
same inputs (pre-event RGB as raw 0-255 floats; post-event image warped bilinearly to the
pre-event grid, as the labels are made on the pre-event image) and the same targets. Changes:

* paths come from committed split files relative to ``$DATA_ROOT`` (``splits/sn8_*.csv``)
* optional random ``crop x crop`` windows (same window for images and masks) for 6 GB GPUs
* returns a dict and builds the flood class map (0 background, 1 dry building,
  2 flooded building, 3 dry road, 4 flooded road) the way ``train_flood.py`` did

Facts checked on real tiles: pre-event 3x1300x1300 uint8, EPSG:4326, nodata 255; post-event
3x~786x786 uint8 (other GSD, same ground); annotations GeoJSON with ``flooded`` "yes"/None.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import data_path, repo_root

SN8_DIR = "sn8"
TRAIN_AOIS = ("Germany_Training_Public", "Louisiana-East_Training_Public")
MASK_KINDS = ("building", "road", "roadspeed", "flood")
FLOOD_CLASSES = ("background", "building", "flooded_building", "road", "flooded_road")


def tile_id(image_name: str) -> str:
    """``10500500C4DD7000_0_41_59.tif`` -> ``0_41_59``."""
    return "_".join(Path(image_name).stem.split("_")[1:])


def rows_from_mapping(aoi: str) -> list[dict[str, str]]:
    """One row per labelled tile of an area, from its ``label_image_mapping.csv``."""
    mapping = pd.read_csv(data_path(SN8_DIR, aoi, f"{aoi}_label_image_mapping.csv"))
    rows = []
    for rec in mapping.to_dict("records"):
        pre = str(rec["pre-event image"])
        post2 = rec.get("post-event image 2")
        stem = Path(pre).stem
        row = {
            "aoi": aoi,
            "event": aoi.split("_")[0],
            "tile_id": tile_id(pre),
            "preimg": f"{SN8_DIR}/{aoi}/PRE-event/{pre}",
            "postimg": f"{SN8_DIR}/{aoi}/POST-event/{rec['post-event image 1']}",
            "postimg2": (
                f"{SN8_DIR}/{aoi}/POST-event/{post2}" if isinstance(post2, str) and post2 else ""
            ),
            "label": f"{SN8_DIR}/{aoi}/annotations/{rec['label']}",
        }
        for kind in MASK_KINDS:
            row[kind] = f"{SN8_DIR}/{aoi}/annotations/masks/{kind}_{stem}.tif"
        rows.append(row)
    return rows


def read_sn8_split(name: str) -> pd.DataFrame:
    """Read ``splits/<name>.csv`` (e.g. ``sn8_train``)."""
    path = repo_root() / "splits" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return pd.read_csv(path, keep_default_na=False)


def read_image(path: Path) -> np.ndarray:
    """All bands of a GeoTIFF as float32 ``(C, H, W)``."""
    from osgeo import gdal

    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()
    ds = None
    arr = np.asarray(arr, dtype=np.float32)
    return arr[None] if arr.ndim == 2 else arr


def warp_post_to_pre(post_path: Path, height: int, width: int) -> np.ndarray:
    """Post-event image resampled to the pre-event pixel grid (baseline ``get_warped_ds``)."""
    from osgeo import gdal

    ds = gdal.Warp(
        "",
        str(post_path),
        format="MEM",
        width=width,
        height=height,
        resampleAlg=gdal.GRIORA_Bilinear,
        outputType=gdal.GDT_Byte,
    )
    arr = ds.ReadAsArray().astype(np.float32)
    ds = None
    return arr


def flood_class_map(flood: np.ndarray) -> np.ndarray:
    """4-channel flood mask -> class index ``(H, W)`` (prepend a zero channel, argmax)."""
    stacked = np.concatenate([np.zeros((1, *flood.shape[1:]), flood.dtype), flood], axis=0)
    return np.argmax(stacked, axis=0).astype(np.int64)


def write_like_pre(path: Path, pre_path: Path, bands: np.ndarray) -> None:
    """Write uint8 bands with the pre-event image's georeferencing (baseline ``write_geotiff``)."""
    from osgeo import gdal

    ref = gdal.Open(str(pre_path))
    driver = gdal.GetDriverByName("GTiff")
    # uncompressed, as the baseline: infer_speed.py reads these with skimage/tifffile
    out = driver.Create(str(path), ref.RasterXSize, ref.RasterYSize, bands.shape[0], gdal.GDT_Byte)
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    for i, band in enumerate(bands, start=1):
        out.GetRasterBand(i).WriteArray(band.astype(np.uint8))
    out.FlushCache()
    out = None
    ref = None
