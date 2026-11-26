"""New-water masks (post and not pre) for the Nepal replay and the event pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import scripts.nepal_level1 as nl
from src.level1_radar.new_water import NODATA, new_water, pre_scene_for, write_new_water_mask

N = 16
TRANSFORM = from_origin(87.0, 26.6, 0.001, 0.001)


def _tif(path: Path, data: np.ndarray, nodata: float | int | None = None) -> Path:
    data = data if data.ndim == 3 else data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=N, width=N, count=data.shape[0], dtype=data.dtype,
        crs="EPSG:4326", transform=TRANSFORM, nodata=nodata,
    ) as dst:  # fmt: skip
        dst.write(data)
    return path


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def test_new_water_is_post_and_not_pre() -> None:
    post = np.array([[0.9, 0.9, 0.1, np.nan]], np.float32)
    pre = np.array([[0.9, 0.1, 0.9, 0.1]], np.float32)
    flood, no_data = new_water(post, pre, 0.5)
    assert flood.tolist() == [[False, True, False, False]]
    assert no_data.tolist() == [[False, False, False, True]]


def test_new_water_no_data_where_pre_missing() -> None:
    flood, no_data = new_water(np.array([0.9], np.float32), np.array([np.nan], np.float32))
    assert not flood[0] and no_data[0]


def test_pre_scene_for(tmp_path: Path) -> None:
    post = tmp_path / "S1_20240927.tif"
    assert pre_scene_for(post) is None
    (tmp_path / "S1_20240927_pre.tif").touch()
    assert pre_scene_for(post).name == "S1_20240927_pre.tif"


def test_write_new_water_mask(tmp_path: Path) -> None:
    post = np.zeros((N, N), np.float32)
    post[:, :8] = 0.9
    pre = np.zeros((N, N), np.float32)
    pre[:4, :] = 0.9  # river channel: water before and after
    pre[15, 0] = np.nan
    out = write_new_water_mask(
        _tif(tmp_path / "post.tif", post),
        _tif(tmp_path / "pre.tif", pre),
        tmp_path / "new.tif",
        block=5,  # several blocks, including partial ones
    )
    mask = _read(out)
    assert mask[:4].max() == 0  # channel is not new water
    assert (mask[4:15, :8] == 1).all() and (mask[4:, 8:] == 0).all()
    assert mask[15, 0] == NODATA


def test_write_new_water_mask_refuses_other_grids(tmp_path: Path) -> None:
    a = _tif(tmp_path / "a.tif", np.zeros((N, N), np.float32))
    with rasterio.open(
        tmp_path / "b.tif", "w", driver="GTiff", height=N, width=N, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(88.0, 26.6, 0.001, 0.001),
    ) as dst:  # fmt: skip
        dst.write(np.zeros((1, N, N), np.float32))
    with pytest.raises(ValueError, match="different grids"):
        write_new_water_mask(a, tmp_path / "b.tif", tmp_path / "out.tif")


def _replay(tmp_path: Path, monkeypatch, with_pre_prob: bool) -> tuple[dict, Path]:
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data))
    rng = np.random.default_rng(0)
    scene = rng.uniform(-25, -5, (2, N, N)).astype(np.float32)
    s1, level1 = data / "nepal/s1", data / "nepal/level1"
    _tif(s1 / "S1_20240927.tif", scene)
    _tif(s1 / "S1_20240927_pre.tif", scene + 1.0)
    jrc = np.zeros((N, N), np.uint8)
    jrc[:, 15] = 100  # permanent water column
    _tif(s1 / "JRC_occurrence.tif", jrc)
    post_prob = np.zeros((N, N), np.float32)
    post_prob[:, :10] = 0.9
    post_prob[:, 15] = 0.9
    _tif(level1 / "S1_20240927_flood_prob.tif", post_prob)
    if with_pre_prob:
        pre_prob = np.zeros((N, N), np.float32)
        pre_prob[:, :3] = 0.9  # normal channel
        _tif(level1 / "S1_20240927_pre_flood_prob.tif", pre_prob)
    cfg = {
        "paths": {
            "s1_dir": "nepal/s1",
            "jrc": "nepal/s1/JRC_occurrence.tif",
            "terrain_dir": "nepal/terrain",
            "level1_dir": "nepal/level1",
        },
        "dates": {"post": ["2024-09-27"]},
        "permanent_water_occurrence": 50,
        "change_otsu": {"min_drop_db": 3.0, "max_hand_m": 15.0},
        "model": {"threshold": 0.5},
    }
    return cfg, level1


def test_masks_write_all_water_and_new_water(tmp_path: Path, monkeypatch) -> None:
    cfg, level1 = _replay(tmp_path, monkeypatch, with_pre_prob=True)
    nl.step_masks(cfg, False, "2024-09-27")
    all_water = _read(level1 / "S1_20240927_model_mask.tif")
    new = _read(level1 / "S1_20240927_model_new_mask.tif")
    assert (all_water[:, :10] == 1).all() and (all_water[:, 15] == 0).all()  # JRC removed
    assert (new[:, :3] == 0).all() and (new[:, 3:10] == 1).all() and (new[:, 15] == 0).all()


def test_masks_skip_new_water_without_pre_prediction(tmp_path: Path, monkeypatch) -> None:
    cfg, level1 = _replay(tmp_path, monkeypatch, with_pre_prob=False)
    nl.step_masks(cfg, False, "2024-09-27")
    assert (level1 / "S1_20240927_model_mask.tif").exists()
    assert not (level1 / "S1_20240927_model_new_mask.tif").exists()  # no empty "all dry" file
