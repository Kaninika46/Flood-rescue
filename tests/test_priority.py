"""Exposure and ranking on synthetic rasters and a toy graph (prompts C2, C3)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from src.common.raster import write_raster
from src.priority.exposure import exposure_by_area
from src.priority.rank import rank_areas
from src.routing.graph import add_edge_ids
from src.routing.route import load_vehicles

UTM = "EPSG:32645"


def _areas() -> gpd.GeoDataFrame:
    # two 1 km x 1 km wards side by side in UTM 45N
    west = gpd.GeoDataFrame(
        {"ward": ["west", "east"]},
        geometry=[box(500000, 2950000, 501000, 2951000), box(501000, 2950000, 502000, 2951000)],
        crs=UTM,
    )
    return west.to_crs("EPSG:4326")


@pytest.fixture()
def rasters(tmp_path: Path) -> tuple[Path, Path, Path]:
    # population: 100 m cells, 10 people each, over both wards (20 x 10 cells)
    pop = np.full((10, 20), 10.0, np.float32)
    write_raster(
        tmp_path / "pop.tif",
        pop,
        crs=UTM,
        transform=Affine(100, 0, 500000, 0, -100, 2951000),
        nodata=-99999.0,
    )
    # flood probability at 10 m: the west ward's left 600 m flooded (0.9), the rest dry (0.1)
    prob = np.full((100, 200), 0.1, np.float32)
    prob[:, :60] = 0.9
    prob[:, 55:60] = 0.4  # a 50 m strip below the threshold inside cell column 5
    write_raster(
        tmp_path / "prob.tif",
        prob,
        crs=UTM,
        transform=Affine(10, 0, 500000, 0, -10, 2951000),
        nodata=float("nan"),
    )
    mask = (prob >= 0.5).astype(np.uint8)
    write_raster(
        tmp_path / "mask.tif",
        mask,
        crs=UTM,
        transform=Affine(10, 0, 500000, 0, -10, 2951000),
        nodata=255,
    )
    return tmp_path / "pop.tif", tmp_path / "prob.tif", tmp_path / "mask.tif"


def test_exposure_majority_and_fraction(rasters: tuple[Path, Path, Path]) -> None:
    pop, prob, mask = rasters
    maj = exposure_by_area(prob, pop, _areas(), "ward").set_index("area_name")
    # cells 0-4 fully flooded, cell 5 is 50 % flooded -> counts under "majority"
    assert maj.loc["west", "exposed_people"] == pytest.approx(10 * 10 * 6)
    assert maj.loc["east", "exposed_people"] == 0
    assert maj.loc["west", "flooded_km2"] == pytest.approx(0.55, abs=0.01)
    assert maj.loc["west", "population"] == pytest.approx(1000)
    frac = exposure_by_area(prob, pop, _areas(), "ward", method="fraction").set_index("area_name")
    assert frac.loc["west", "exposed_people"] == pytest.approx(550, abs=5)
    from_mask = exposure_by_area(mask, pop, _areas(), "ward").set_index("area_name")
    assert from_mask.loc["west", "exposed_people"] == maj.loc["west", "exposed_people"]


def _toy_graph() -> nx.MultiDiGraph:
    # hospital H at the west; west ward node W reached only through a flooded edge;
    # east ward node E reached through dry roads
    areas = _areas()
    cw, ce = areas.geometry.iloc[0].centroid, areas.geometry.iloc[1].centroid
    G = nx.MultiDiGraph(crs="EPSG:4326")
    G.add_node("H", x=cw.x - 0.02, y=cw.y)
    G.add_node("W", x=cw.x, y=cw.y)
    G.add_node("E", x=ce.x, y=ce.y)
    G.add_node("J", x=cw.x - 0.01, y=cw.y + 0.02)
    edges = [
        ("H", "W", {"status": "flooded", "max_depth_m": 0.8}),
        ("H", "J", {"status": "dry"}),
        ("J", "E", {"status": "dry"}),
    ]
    for u, v, d in edges:
        for a, b in ((u, v), (v, u)):
            G.add_edge(a, b, key=0, length=100.0, travel_time=10.0, **d)
    return add_edge_ids(G)


def test_rank_puts_cut_off_first(rasters: tuple[Path, Path, Path]) -> None:
    pop, prob, _ = rasters
    exposure = exposure_by_area(prob, pop, _areas(), "ward")
    G = _toy_graph()
    hospital = (G.nodes["H"]["x"], G.nodes["H"]["y"])
    ranked = rank_areas(exposure, G, [hospital], load_vehicles())
    assert list(ranked["area_name"]) == ["west", "east"]
    assert ranked.loc[0, "cut_off"] and not ranked.loc[1, "cut_off"]
    assert list(ranked["priority"]) == [1, 2]


def test_rank_by_exposed_when_nothing_cut_off(rasters: tuple[Path, Path, Path]) -> None:
    pop, prob, _ = rasters
    exposure = exposure_by_area(prob, pop, _areas(), "ward")
    G = _toy_graph()
    for _, _, d in G.edges(data=True):
        d["status"], d["max_depth_m"] = "dry", float("nan")
    hospital = (G.nodes["H"]["x"], G.nodes["H"]["y"])
    ranked = rank_areas(exposure, G, [hospital], load_vehicles())
    assert not ranked["cut_off"].any()
    assert ranked.loc[0, "area_name"] == "west"  # more exposed people
