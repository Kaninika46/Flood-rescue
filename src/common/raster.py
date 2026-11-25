"""Read and write GeoTIFFs without losing the grid (CRS, transform) or the nodata value.

Arrays are always ``(bands, height, width)``. A single-band raster is ``(1, H, W)``;
:func:`write_raster` also accepts a 2-D array for one band.

Example::

    s1 = read_raster(data_path("sen1floods11/.../Bolivia_103757_S1Hand.tif"))
    vv, vh = s1.array
    write_like(out_path, prob[None].astype("float32"), s1, nodata=float("nan"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

PathLike = str | os.PathLike[str]


@dataclass
class Raster:
    """A raster in memory: pixel array plus the georeferencing needed to write it back."""

    array: np.ndarray
    crs: CRS | None
    transform: Affine
    nodata: float | int | None
    band_names: tuple[str | None, ...] = field(default=())

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)."""
        return int(self.array.shape[-2]), int(self.array.shape[-1])

    @property
    def count(self) -> int:
        """Number of bands."""
        return int(self.array.shape[0])

    def valid_mask(self) -> np.ndarray:
        """Boolean ``(H, W)`` mask: True where no band is nodata (or NaN)."""
        invalid = np.zeros(self.shape, dtype=bool)
        if self.array.dtype.kind == "f":
            invalid |= np.isnan(self.array).any(axis=0)
        if self.nodata is not None and not (
            isinstance(self.nodata, float) and np.isnan(self.nodata)
        ):
            invalid |= (self.array == self.nodata).any(axis=0)
        return ~invalid


def read_raster(
    path: PathLike,
    bands: int | list[int] | None = None,
    window: Window | None = None,
) -> Raster:
    """Read a raster file into a :class:`Raster`.

    Args:
        path: GeoTIFF (or any GDAL-readable raster).
        bands: 1-based band index or list of indices; ``None`` reads all bands.
        window: Optional ``rasterio.windows.Window`` to read part of the file.
            The returned transform is the window's transform.
    """
    indexes = [bands] if isinstance(bands, int) else bands
    with rasterio.open(path) as src:
        array = src.read(indexes=indexes, window=window)
        transform = src.window_transform(window) if window is not None else src.transform
        all_names = src.descriptions
        chosen = indexes if indexes is not None else list(range(1, src.count + 1))
        names = tuple(all_names[i - 1] for i in chosen)
        return Raster(
            array=array, crs=src.crs, transform=transform, nodata=src.nodata, band_names=names
        )


def describe_raster(path: PathLike) -> dict[str, Any]:
    """Shape, dtype, CRS, nodata and per-band value range (ignoring nodata and NaN).

    Use it before writing any loader for a new file type.
    """
    raster = read_raster(path)
    valid = raster.valid_mask()
    ranges = []
    for band in raster.array:
        values = band[valid]
        if values.size:
            ranges.append((float(values.min()), float(values.max())))
        else:
            ranges.append((None, None))
    return {
        "path": str(path),
        "shape": tuple(raster.array.shape),
        "dtype": str(raster.array.dtype),
        "crs": raster.crs.to_string() if raster.crs else None,
        "transform": tuple(raster.transform)[:6],
        "nodata": raster.nodata,
        "band_names": raster.band_names,
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "band_ranges": ranges,
    }


def write_raster(
    path: PathLike,
    array: np.ndarray,
    *,
    crs: CRS | str | None,
    transform: Affine,
    nodata: float | int | None = None,
    band_names: list[str] | tuple[str, ...] | None = None,
    compress: str = "deflate",
) -> Path:
    """Write a tiled, compressed GeoTIFF and return its path.

    Args:
        path: Output file; parent folders are created.
        array: ``(bands, H, W)`` or ``(H, W)``. The array's dtype is written as is.
        crs: Coordinate reference system (``CRS`` or a string like ``"EPSG:4326"``).
        transform: Affine geotransform of the top-left pixel.
        nodata: Nodata value stored in the file (use ``float("nan")`` for float rasters).
        band_names: Optional description per band, e.g. ``["VV", "VH"]``.
        compress: GDAL compression (``deflate``, ``lzw`` or ``none``).
    """
    if array.ndim == 2:
        array = array[np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"array must be (bands, H, W) or (H, W), got shape {array.shape}")
    if array.dtype == bool:
        array = array.astype(np.uint8)
    if band_names is not None and len(band_names) != array.shape[0]:
        raise ValueError(f"{len(band_names)} band names for {array.shape[0]} bands")
    count, height, width = array.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    if compress and compress.lower() != "none":
        profile["compress"] = compress
        if array.dtype.kind == "f":
            profile["predictor"] = 3
        elif array.dtype.kind in "iu":
            profile["predictor"] = 2
    if height >= 256 and width >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        if band_names is not None:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
    return path


def write_like(
    path: PathLike,
    array: np.ndarray,
    template: Raster,
    *,
    nodata: float | int | None = None,
    band_names: list[str] | tuple[str, ...] | None = None,
    compress: str = "deflate",
) -> Path:
    """Write ``array`` on exactly the same grid (CRS, transform, size) as ``template``.

    ``nodata`` defaults to the template's nodata. Raises if the height/width differ.
    """
    out_shape = array.shape[-2:]
    if tuple(out_shape) != template.shape:
        raise ValueError(f"array grid {tuple(out_shape)} != template grid {template.shape}")
    return write_raster(
        path,
        array,
        crs=template.crs,
        transform=template.transform,
        nodata=template.nodata if nodata is None else nodata,
        band_names=band_names,
        compress=compress,
    )
