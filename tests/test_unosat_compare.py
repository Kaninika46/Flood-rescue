"""UNOSAT layer selection and the date-matched Nepal comparison (tiny synthetic rasters)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import scripts.nepal_level1 as nl
from src.common.unosat import CREDIT, file_date, find_layers, layer_kind

# 20 x 20 grid of 0.001 degree pixels; column c spans X0 + c*RES, row r spans Y0 - r*RES
X0, Y0, RES, N = 87.0, 26.6, 0.001, 20
TRANSFORM = from_origin(X0, Y0, RES, RES)


def _cells(c0: int, c1: int, r0: int, r1: int):
    """Polygon covering columns c0..c1-1 and rows r0..r1-1."""
    return box(X0 + c0 * RES, Y0 - r1 * RES, X0 + c1 * RES, Y0 - r0 * RES)


def _tif(path: Path, data: np.ndarray, nodata: float | int | None = None) -> None:
    data = data if data.ndim == 3 else data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=N, width=N, count=data.shape[0], dtype=data.dtype,
        crs="EPSG:4326", transform=TRANSFORM, nodata=nodata,
    ) as dst:  # fmt: skip
        dst.write(data)


def _vector(path: Path, geom) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326").to_file(path)


# ------------------------------------------------------------------ layer selection


def test_file_date_and_kind() -> None:
    p = Path("S1_20241001_FloodExtent_Koshi_Madhesh.shp")
    assert file_date(p) == "2024-10-01"
    assert layer_kind(p) == "flood"
    assert layer_kind(Path("S1_20241001_AnalysisExtent_Koshi_Madhesh.shp")) == "extent"
    assert file_date(Path("flood.geojson")) is None


def test_find_layers_matches_file_name_date_recursively(tmp_path: Path) -> None:
    sub = tmp_path / "FL20240928NPL_SHP"
    for name in [
        "S1_20240927_FloodExtent_A.shp",
        "S1_20240927_AnalysisExtent_A.shp",
        "S1_20241001_FloodExtent_A.shp",
        "S1_20240929_FloodExtent_B.shp",
    ]:
        _vector(sub / name, _cells(0, 1, 0, 1))
    layers = find_layers(tmp_path, "2024-09-27")
    assert layers.flood.name == "S1_20240927_FloodExtent_A.shp"
    assert layers.extent.name == "S1_20240927_AnalysisExtent_A.shp"
    oct1 = find_layers(tmp_path, "2024-10-01")
    assert oct1.flood.name == "S1_20241001_FloodExtent_A.shp" and oct1.extent is None
    assert find_layers(tmp_path, "2024-09-28").flood is None


def test_find_layers_refuses_two_floods_of_one_date(tmp_path: Path) -> None:
    _vector(tmp_path / "S1_20240927_FloodExtent_A.shp", _cells(0, 1, 0, 1))
    _vector(tmp_path / "S1_20240927_FloodExtent_B.shp", _cells(0, 1, 0, 1))
    with pytest.raises(ValueError, match="several UNOSAT flood layers"):
        find_layers(tmp_path, "2024-09-27")


# ------------------------------------------------------------------ comparison


@pytest.fixture
def replay(tmp_path: Path, monkeypatch) -> dict:
    """Scenes, JRC, a model mask, UNOSAT layers for two dates and a Nepal outline."""
    data, repo = tmp_path / "data", tmp_path / "repo"
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setattr(nl, "repo_root", lambda: repo)
    monkeypatch.setattr(nl, "make_figures", lambda cfg, post: None)

    scene = np.full((2, N, N), -15.0, np.float32)
    scene[:, :, 19] = np.nan  # outside our scene footprint
    for day in ("20240927", "20241001"):
        _tif(data / "nepal/s1" / f"S1_{day}.tif", scene)
    jrc = np.zeros((N, N), np.uint8)
    jrc[0:5, :] = 100  # rows 0-4: permanent water
    _tif(data / "nepal/s1/JRC_occurrence.tif", jrc)
    # model: flood in columns 0-9, but not on the permanent water rows
    mask = np.zeros((N, N), np.uint8)
    mask[5:, 0:10] = 1
    _tif(data / "nepal/level1/S1_20240927_model_mask.tif", mask, nodata=255)

    unosat = data / "nepal/unosat/FL_SHP"
    _vector(unosat / "S1_20240927_FloodExtent_K.shp", _cells(0, 10, 0, 20))  # left half
    _vector(unosat / "S1_20241001_FloodExtent_K.shp", _cells(10, 20, 0, 20))  # right half
    for day in ("20240927", "20241001"):
        _vector(unosat / f"S1_{day}_AnalysisExtent_K.shp", _cells(0, 20, 0, 20))
    admin = data / "nepal/admin/npl.geojson"
    admin.parent.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[_cells(0, 20, 10, 20)], crs="EPSG:4326").to_file(admin)

    cfg = {
        "paths": {
            "s1_dir": "nepal/s1",
            "jrc": "nepal/s1/JRC_occurrence.tif",
            "terrain_dir": "nepal/terrain",
            "level1_dir": "nepal/level1",
            "unosat_dir": "nepal/unosat",
            "admin": "nepal/admin/npl.geojson",
        },
        "dates": {"post": ["2024-09-27", "2024-10-01"]},
        "permanent_water_occurrence": 50,
        "unosat": {
            "product": "FL20240928NPL",
            "reference": {
                "2024-09-27": {
                    "acquired_utc": "00:11",
                    "analysed_km2": 1,
                    "flood_km2": 1,
                    "exposed": 17317,
                }
            },
        },
    }
    return {"cfg": cfg, "repo": repo, "data": data}


def test_compare_uses_same_date_and_leaves_out_permanent_water(replay: dict) -> None:
    nl.step_compare(replay["cfg"], False, "2024-09-27")
    res = json.loads((replay["repo"] / "docs/results/nepal_level1_20240927.json").read_text())
    assert res["unosat_flood_layer"] == "S1_20240927_FloodExtent_K.shp"
    assert res["credit"] == CREDIT
    whole = res["regions"]["unosat_extent"]
    # same-date reference and permanent water removed from both maps -> perfect agreement
    assert whole["methods"]["model"]["iou"] == pytest.approx(1.0)
    # UNOSAT polygons on permanent-water rows are counted only in the "before" figure
    assert whole["unosat_flood_km2_before_jrc_removal"] > whole["unosat_flood_km2"]
    nepal = res["regions"]["nepal"]
    # Nepal part: rows 10-19 of rows 5-19 with data, so two thirds of the analysed area
    assert nepal["analysed_km2"] / whole["analysed_km2"] == pytest.approx(2 / 3, rel=0.05)
    doc = (replay["repo"] / "docs/nepal_level1.md").read_text()
    assert "Analysis: UNOSAT" in doc and "## 2024-09-27" in doc and "2024-10-01" not in doc


def test_compare_against_another_dates_extent_disagrees(replay: dict) -> None:
    """The 27 Sep mask scored as if it were 1 Oct must not agree (guards the date matching)."""
    data = replay["data"]
    (data / "nepal/level1/S1_20240927_model_mask.tif").rename(
        data / "nepal/level1/S1_20241001_model_mask.tif"
    )
    nl.step_compare(replay["cfg"], False, "2024-10-01")
    res = json.loads((replay["repo"] / "docs/results/nepal_level1_20241001.json").read_text())
    assert res["unosat_flood_layer"] == "S1_20241001_FloodExtent_K.shp"
    assert res["regions"]["unosat_extent"]["methods"]["model"]["iou"] == 0.0


def test_terrain_file_reuses_same_grid_terrain(replay: dict) -> None:
    cfg, data = replay["cfg"], replay["data"]
    hand = data / "nepal/terrain/S1_20240927_HAND.tif"
    _tif(hand, np.zeros((N, N), np.float32))
    post_oct = data / "nepal/s1/S1_20241001.tif"
    assert nl.terrain_file(cfg, post_oct, "hand") == hand
    assert nl.terrain_file(cfg, post_oct, "slope").name == "S1_20241001_SLOPE.tif"
