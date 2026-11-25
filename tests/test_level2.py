"""Level 2 helpers on synthetic data (no SpaceNet 8 files, no GPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from src.level2_roads.cloud import cloud_mask
from src.level2_roads.data import FLOOD_CLASSES, flood_class_map, tile_id
from src.level2_roads.road_status import decide
from src.level2_roads.scoring import is_flooded, length_precision_recall, lon_lat_order


def test_tile_id_and_flood_classes() -> None:
    assert tile_id("10500500C4DD7000_0_41_59.tif") == "0_41_59"
    flood = np.zeros((4, 2, 3), np.uint8)
    flood[1, 0, 0] = 1  # flooded building
    flood[3, 1, 2] = 1  # flooded road
    flood[2, 1, 0] = 1  # dry road
    classes = flood_class_map(flood)
    assert FLOOD_CLASSES[classes[0, 0]] == "flooded_building"
    assert FLOOD_CLASSES[classes[1, 2]] == "flooded_road"
    assert FLOOD_CLASSES[classes[1, 0]] == "road"
    assert classes[0, 1] == 0


def test_length_precision_recall_partial_overlap() -> None:
    truth = [LineString([(0, 0), (100, 0)])]
    pred = [LineString([(50, 1), (150, 1)])]  # 1 m off, overlapping from x=50 to 100
    r = length_precision_recall(pred, truth, buffer_m=5)
    # the 5 m buffers reach 5 m past each line's end: 55 m of 100 m covered on both sides
    assert r["precision"] == pytest.approx(0.55, abs=0.01)
    assert r["recall"] == pytest.approx(0.55, abs=0.01)
    assert r["segment_recall"] == 1.0


def test_length_precision_recall_empty_sides() -> None:
    r = length_precision_recall([], [LineString([(0, 0), (10, 0)])])
    assert r["recall"] == 0.0 and math.isnan(r["precision"])
    r = length_precision_recall([LineString([(0, 0), (10, 0)])], [])
    assert r["precision"] == 0.0 and math.isnan(r["recall"])


def test_flood_flags() -> None:
    assert is_flooded("yes") and is_flooded("True") and is_flooded(True)
    assert not is_flooded("no") and not is_flooded(None) and not is_flooded(float("nan"))
    assert not is_flooded("Null")


def test_lat_lon_axis_order_is_fixed() -> None:
    bounds = (-90.03, 29.82, -90.02, 29.83)  # lon/lat tile footprint
    lat_lon = LineString([(29.825, -90.025), (29.826, -90.024)])
    fixed = lon_lat_order(lat_lon, bounds)
    assert fixed.coords[0] == (-90.025, 29.825)
    lon_lat = LineString([(-90.025, 29.825), (-90.024, 29.826)])
    assert lon_lat_order(lon_lat, bounds) == lon_lat


def test_cloud_mask_bright_smooth_only() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(40, 120, (3, 120, 120)).astype(np.uint8)  # textured ground
    img[:, :60, :60] = 230  # flat bright cloud
    roof = rng.integers(150, 255, (3, 40, 40)).astype(np.uint8)  # bright but textured roof
    img[:, 70:110, 70:110] = roof
    mask = cloud_mask(img, {"min_area_px": 50})
    assert mask[10:50, 10:50].mean() > 0.95  # border pixels see the texture of the ground
    assert mask[75:105, 75:105].mean() < 0.05
    assert mask[:, 115:].sum() == 0


def test_status_rules() -> None:
    cfg = {"cloud": {"min_fraction": 0.5}, "ensemble": {"verify_disagreement": 0.34}}
    assert decide(True, 0.8, 0.0, cfg) == ("unknown", 0.0)
    assert decide(True, 0.1, 0.5, cfg)[0] == "verify"
    assert decide(True, 0.1, 0.0, cfg) == ("flooded", 1.0)
    assert decide(False, 0.0, float("nan"), cfg) == ("dry", 1.0)
