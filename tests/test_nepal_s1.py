"""Nepal Sentinel-1 export planning and scene pairing (no Earth Engine, no real data)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from scripts.nepal_level1 import scenes
from scripts.nepal_s1_download import (
    METRES_PER_DEGREE,
    estimate_bytes,
    export_shape,
    human_size,
    pair_pre_scenes,
    select_post_dates,
)


def _row(date: str, orbit: int) -> dict[str, object]:
    return {"date": date, "orbit": orbit, "pass": "X", "id": f"{date}-{orbit}"}


PRE = [
    _row("2024-09-03", 121),
    _row("2024-09-07", 12),
    _row("2024-09-12", 85),
    _row("2024-09-15", 121),
    _row("2024-09-19", 12),
]


def test_each_post_date_gets_latest_pre_on_its_own_orbit() -> None:
    post = {"2024-09-27": [_row("2024-09-27", 121)], "2024-10-01": [_row("2024-10-01", 12)]}
    pairs = pair_pre_scenes(post, PRE)
    assert pairs["2024-09-27"]["date"] == "2024-09-15"
    assert pairs["2024-09-27"]["orbit"] == 121
    assert pairs["2024-10-01"]["date"] == "2024-09-19"
    assert pairs["2024-10-01"]["orbit"] == 12


def test_no_pre_scene_on_orbit_gives_none() -> None:
    pairs = pair_pre_scenes({"2024-09-27": [_row("2024-09-27", 48)]}, PRE)
    assert pairs["2024-09-27"] is None


def test_pre_scene_must_be_before_post_date() -> None:
    pairs = pair_pre_scenes({"2024-09-10": [_row("2024-09-10", 121)]}, PRE)
    assert pairs["2024-09-10"]["date"] == "2024-09-03"


def test_select_post_dates() -> None:
    post = ["2024-09-27", "2024-10-01"]
    assert select_post_dates(post, None) == post
    assert select_post_dates(post, "2024-10-01") == ["2024-10-01"]
    with pytest.raises(typer.BadParameter):
        select_post_dates(post, "2024-09-28")


def test_export_shape_geographic() -> None:
    step = 10 / METRES_PER_DEGREE
    width, height = export_shape((85.0, 26.0, 85.0 + 100 * step, 26.0 + 50 * step), 10, "EPSG:4326")
    assert (width, height) == (100, 50)


def test_export_shape_projected_metres() -> None:
    # About 1 km x 1 km near the equator in UTM 31N at 10 m.
    width, height = export_shape((3.0, 0.0, 3.009, 0.009), 10, "EPSG:32631")
    assert 95 <= width <= 105 and 95 <= height <= 105


def test_estimate_bytes_and_human_size() -> None:
    assert estimate_bytes(1000, 500, 2) == 4_000_000
    assert human_size(4_000_000) == "4.0 MB"
    assert human_size(12_300_000_000) == "12.3 GB"


def _cfg(post: list[str]) -> dict[str, object]:
    return {"paths": {"s1_dir": "nepal/s1"}, "dates": {"post": post}}


def test_scenes_uses_pre_named_after_post_date(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    s1 = tmp_path / "nepal" / "s1"
    s1.mkdir(parents=True)
    for name in ["S1_20240927.tif", "S1_20240927_pre.tif", "S1_20241001_pre.tif"]:
        (s1 / name).touch()
    post, pre = scenes(_cfg(["2024-09-27", "2024-10-01"]))
    assert post.name == "S1_20240927.tif"
    assert pre is not None and pre.name == "S1_20240927_pre.tif"


def test_scenes_ignores_other_dates_pre(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    s1 = tmp_path / "nepal" / "s1"
    s1.mkdir(parents=True)
    (s1 / "S1_20240927.tif").touch()
    (s1 / "S1_20241001_pre.tif").touch()  # newest *_pre, but for another post date
    post, pre = scenes(_cfg(["2024-09-27"]))
    assert post.name == "S1_20240927.tif"
    assert pre is None
