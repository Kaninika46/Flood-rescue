"""Map (a): all water minus JRC permanent water; and the (c) sensitivity mask in the replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import scripts.nepal_level1 as nl
from src.level1_radar.flood_mask import NODATA, all_water, write_all_water_mask

N = 16


def _tif(path: Path, data: np.ndarray, origin=(87.0, 26.6), res=0.001) -> Path:
    data = data if data.ndim == 3 else data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=data.dtype, crs="EPSG:4326",
        transform=from_origin(*origin, res, res),
    ) as dst:  # fmt: skip
        dst.write(data)
    return path


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def test_all_water_removes_only_permanent_jrc() -> None:
    prob = np.array([[0.9, 0.9, 0.9, 0.1, np.nan]], np.float32)
    jrc = np.array([[0, 30, 80, 0, 0]], np.float32)
    flood, no_data = all_water(prob, jrc, 0.5, 50)
    assert flood.tolist() == [[True, True, False, False, False]]  # seasonal (30 %) is kept
    assert no_data.tolist() == [[False, False, False, False, True]]


def test_all_water_without_jrc_and_outside_jrc() -> None:
    prob = np.array([[0.9, 0.9]], np.float32)
    assert all_water(prob, None)[0].all()
    assert all_water(prob, np.array([[np.nan, 90]], np.float32))[0].tolist() == [[True, False]]


def test_write_all_water_mask_reprojects_jrc(tmp_path: Path) -> None:
    prob = np.full((N, N), 0.9, np.float32)
    prob[0, 0] = np.nan
    # JRC on a coarser grid (2x pixels) covering the same area: left half permanent
    jrc = np.zeros((N // 2, N // 2), np.uint8)
    jrc[:, : N // 4] = 100
    out = write_all_water_mask(
        _tif(tmp_path / "prob.tif", prob),
        tmp_path / "mask.tif",
        jrc_path=_tif(tmp_path / "jrc.tif", jrc, res=0.002),
        block=5,
    )
    mask = _read(out)
    assert mask[0, 0] == NODATA
    assert (mask[1:, : N // 2] == 0).all() and (mask[:, N // 2 :] == 1).all()


def test_replay_writes_sensitivity_mask(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data))
    rng = np.random.default_rng(0)
    _tif(data / "nepal/s1/S1_20240927.tif", rng.uniform(-25, -5, (2, N, N)).astype(np.float32))
    jrc = np.zeros((N, N), np.uint8)
    jrc[:, 5] = 20  # seasonal
    jrc[:, 15] = 100  # permanent
    _tif(data / "nepal/s1/JRC_occurrence.tif", jrc)
    prob = np.zeros((N, N), np.float32)
    prob[:, :10] = 0.9
    prob[:, 15] = 0.9
    _tif(data / "nepal/level1/S1_20240927_flood_prob.tif", prob)
    cfg = {
        "paths": {
            "s1_dir": "nepal/s1",
            "jrc": "nepal/s1/JRC_occurrence.tif",
            "terrain_dir": "nepal/terrain",
            "level1_dir": "nepal/level1",
        },
        "dates": {"post": ["2024-09-27"]},
        "permanent_water_occurrence": 50,
        "sensitivity_jrc_occurrence": 10,
        "change_otsu": {"min_drop_db": 3.0, "max_hand_m": 15.0},
        "model": {"threshold": 0.5},
    }
    nl.step_masks(cfg, False, "2024-09-27")
    a = _read(data / "nepal/level1/S1_20240927_model_mask.tif")
    c = _read(data / "nepal/level1/S1_20240927_model_jrc10_mask.tif")
    assert (a[:, 5] == 1).all() and (a[:, 15] == 0).all()  # (a) keeps seasonal water
    assert (c[:, 5] == 0).all() and (c[:, 4] == 1).all()  # (c) drops it
