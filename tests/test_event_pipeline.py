"""run_event uses the replay's flood and depth basis: map (a), permanent water excluded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.event_pipeline as ep


def test_depth_uses_map_a_mask_not_raw_probability(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_depth(flood_path, settings, **kwargs):
        seen["path"], seen["kwargs"] = flood_path, kwargs
        return np.zeros((2, 2), np.float32), type("G", (), {"crs": "EPSG:32645", "transform": None})

    monkeypatch.setattr("src.depth.depth_map.depth_from_flood", fake_depth)
    monkeypatch.setattr("src.common.raster.write_raster", lambda *a, **k: None)
    run = ep.EventRun(scene=tmp_path / "S1_x.tif", out_dir=tmp_path)
    run.outputs.update(prob=tmp_path / "prob.tif", mask=tmp_path / "mask.tif")
    ep.step_depth(run)
    assert seen["path"] == tmp_path / "mask.tif"  # river under bridges stays dry
    assert seen["kwargs"]["is_mask"] is True


def test_run_event_builds_mask_before_depth(monkeypatch, tmp_path: Path) -> None:
    order = []
    for name in ("level1", "mask", "depth", "level2", "roads", "priority", "alert"):
        monkeypatch.setattr(
            ep, f"step_{name}", lambda *a, _n=name, **k: order.append(_n), raising=True
        )
    ep.run_event(tmp_path / "S1_x.tif", tmp_path / "out")
    assert order.index("mask") < order.index("depth") < order.index("roads")


def test_alert_identifier_unique_per_run() -> None:
    from datetime import UTC, datetime

    scene = Path("S1_20240927.tif")
    a = ep.alert_identifier(scene, datetime(2026, 11, 26, 16, 0, 1, tzinfo=UTC))
    b = ep.alert_identifier(scene, datetime(2026, 11, 26, 16, 0, 2, tzinfo=UTC))
    assert a == "S1_20240927-20261126T160001Z-p1" and a != b


def test_alert_record_id_matches_cap_identifier(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    ranked = gpd.GeoDataFrame(
        {"area_name": ["Barah"]}, geometry=[box(87.0, 26.5, 87.1, 26.6)], crs="EPSG:4326"
    )
    record = ep.step_alert(ep.EventRun(scene=tmp_path / "S1_x.tif", out_dir=tmp_path), ranked)
    assert f"<identifier>{record.alert_id}</identifier>" in record.cap_xml
    assert "upper estimate" in record.cap_xml
