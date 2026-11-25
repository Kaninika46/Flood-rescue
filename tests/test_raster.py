"""Tests for src/common/raster.py: a synthetic GeoTIFF keeps its grid and nodata."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

from src.common.raster import describe_raster, read_raster, write_like, write_raster  # noqa: E402

UTM_45N = "EPSG:32645"  # Koshi / Madhesh
TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2950000.0)


def test_round_trip_keeps_crs_transform_nodata_and_values(tmp_path: Path) -> None:
    array = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) - 20.0  # dB-like values
    array[0, 0, 0] = -9999.0
    path = write_raster(
        tmp_path / "sub" / "s1.tif",
        array,
        crs=UTM_45N,
        transform=TRANSFORM,
        nodata=-9999.0,
        band_names=["VV", "VH"],
    )
    r = read_raster(path)
    assert r.crs.to_epsg() == 32645
    assert r.transform == TRANSFORM
    assert r.nodata == -9999.0
    assert r.array.dtype == np.float32
    np.testing.assert_array_equal(r.array, array)
    assert r.band_names == ("VV", "VH")
    assert r.valid_mask().sum() == 4 * 5 - 1


def test_single_band_2d_and_band_selection(tmp_path: Path) -> None:
    label = np.array([[-1, 0], [1, 1]], dtype=np.int16)
    path = write_raster(tmp_path / "label.tif", label, crs=UTM_45N, transform=TRANSFORM, nodata=-1)
    r = read_raster(path, bands=1)
    assert r.array.shape == (1, 2, 2)
    assert r.array.dtype == np.int16
    assert r.nodata == -1


def test_nan_nodata_and_write_like(tmp_path: Path) -> None:
    template = read_raster(
        write_raster(
            tmp_path / "t.tif",
            np.zeros((1, 300, 260), np.float32),
            crs=UTM_45N,
            transform=TRANSFORM,
        )
    )
    prob = np.full((1, 300, 260), 0.25, dtype=np.float32)
    prob[0, :10, :10] = np.nan
    out = read_raster(write_like(tmp_path / "prob.tif", prob, template, nodata=float("nan")))
    assert out.transform == template.transform and out.crs == template.crs
    assert math.isnan(out.nodata)
    assert out.valid_mask().sum() == 300 * 260 - 100
    with pytest.raises(ValueError, match="grid"):
        write_like(tmp_path / "bad.tif", np.zeros((1, 10, 10), np.float32), template)


def test_describe_raster_reports_ranges(tmp_path: Path) -> None:
    array = np.array([[[-25.0, -5.0], [-9999.0, -12.0]]], dtype=np.float32)
    path = write_raster(tmp_path / "d.tif", array, crs=UTM_45N, transform=TRANSFORM, nodata=-9999.0)
    info = describe_raster(path)
    assert info["shape"] == (1, 2, 2)
    assert info["crs"] == "EPSG:32645"
    assert info["band_ranges"] == [(-25.0, -5.0)]
    assert info["valid_fraction"] == pytest.approx(0.75)
