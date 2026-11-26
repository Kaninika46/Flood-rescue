"""One flood event from radar scene to alert draft, step by step, each step timed.

Used by ``scripts/run_event.py`` (D2: stores results in the database) and
``scripts/time_pipeline.py`` (C5: reports minutes per step). Steps:

1. ``level1``   ensemble flood probability + confidence (src/level1_radar/predict.py)
1b. ``mask``   map (a): all water minus JRC permanent water (``configs/nepal.yaml`` paths.jrc,
                when that file exists), used by the priority step. It also counts the normal river
                channel, so exposure is an upper estimate (safety first: src/level1_radar/
                flood_mask.py). The new-water map (b) is not used: it under-counts flood.
2. ``depth``    FwDET depth on a 30 m UTM grid (src/depth)
3. ``level2``   SpaceNet 8 road status, only when Maxar tiles are given (sn8 env; an evaluation
                folder with ``road_status.geojson`` is read)
4. ``roads``    status of every road edge + a demo route (src/routing)
5. ``priority`` exposure per area and the ranked priority list (src/priority)
6. ``alert``    CAP 1.2 draft for the top area (src/alerts); a draft only, never sent here
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger

log = get_logger("event")


@dataclass
class EventRun:
    """Paths and timings of one event run."""

    scene: Path
    out_dir: Path
    outputs: dict[str, Any] = field(default_factory=dict)
    minutes: dict[str, float] = field(default_factory=dict)

    def timed(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run one step and record its wall-clock minutes."""
        start = time.perf_counter()
        log.info("== %s", name)
        result = fn()
        self.minutes[name] = round((time.perf_counter() - start) / 60, 3)
        log.info("%s took %.2f min", name, self.minutes[name])
        return result

    @property
    def total_minutes(self) -> float:
        """Image file to alert draft."""
        return round(sum(self.minutes.values()), 3)


def step_level1(run: EventRun, model_config: str, hand: Path | None) -> None:
    """Flood probability and confidence rasters."""
    from src.level1_radar.predict import Ensemble, member_paths, predict_scene

    ensemble = Ensemble.from_checkpoints(member_paths(load_config(repo_root() / model_config)))
    terrain = {"HAND": hand} if hand is not None and "HAND" in ensemble.channels else {}
    if "HAND" in ensemble.channels and not terrain:
        from src.level1_radar.terrain import write_terrain_rasters

        terrain = {"HAND": write_terrain_rasters(run.scene, run.out_dir, kinds=("hand",))["hand"]}
    prob, std = predict_scene(ensemble, run.scene, run.out_dir, terrain)
    run.outputs.update(prob=prob, std=std)


def step_mask(run: EventRun, threshold: float = 0.5) -> None:
    """Map (a) mask: probability >= threshold minus JRC permanent water."""
    from src.level1_radar.flood_mask import write_all_water_mask

    nepal = load_config(repo_root() / "configs" / "nepal.yaml")
    jrc = data_path(nepal["paths"]["jrc"])
    if not jrc.exists():
        log.warning("%s missing: map (a) keeps permanent water", jrc)
    out = run.out_dir / f"{run.scene.stem}_flood_mask.tif"
    run.outputs["mask"] = write_all_water_mask(
        run.outputs["prob"], out, threshold, jrc, nepal["permanent_water_occurrence"]
    )


def step_depth(run: EventRun, threshold: float = 0.5, resolution_m: float = 30.0) -> None:
    """Depth GeoTIFF from the probability map."""
    from src.common.raster import write_raster
    from src.depth.depth_map import depth_from_flood
    from src.depth.fwdet import FwdetSettings

    settings = FwdetSettings(**load_config(repo_root() / "configs" / "depth.yaml")["fwdet"])
    depth, grid = depth_from_flood(
        run.outputs["prob"], settings, threshold=threshold, resolution_m=resolution_m
    )
    out = run.out_dir / f"{run.scene.stem}_depth.tif"
    write_raster(
        out,
        depth,
        crs=grid.crs,
        transform=grid.transform,
        nodata=float("nan"),
        band_names=["water_depth_m"],
    )
    run.outputs["depth"] = out


def step_level2(run: EventRun, road_status: Path | None) -> None:
    """Attach Level 2 road status when an evaluation folder exists."""
    if road_status is not None and road_status.exists():
        run.outputs["level2_status"] = road_status
    else:
        log.info("no Level 2 road status for this event")


def step_roads(run: EventRun) -> Any:
    """Edge statuses on the OSM graph and one demo route per vehicle."""
    import geopandas as gpd

    from src.routing.graph import load_graph
    from src.routing.route import NoRoute, load_vehicles, route
    from src.routing.status import StatusSettings, assign_status, status_counts

    cfg = load_config(repo_root() / "configs" / "routing.yaml")
    G = load_graph(data_path(cfg["graph"]["path"]))
    level2 = run.outputs.get("level2_status")
    assign_status(
        G,
        depth=run.outputs.get("depth"),
        std=run.outputs.get("std"),
        level2=gpd.read_file(level2) if level2 else None,
        settings=StatusSettings(**cfg["status"]),
    )
    run.outputs["edge_status_counts"] = dict(status_counts(G))
    routes = {}
    for vehicle in cfg["demo"]["vehicles"]:
        try:
            routes[vehicle] = route(
                G, tuple(cfg["demo"]["start"]), tuple(cfg["demo"]["end"]), vehicle, load_vehicles()
            )
        except NoRoute:
            routes[vehicle] = None
    run.outputs["routes"] = routes
    run.outputs["graph"] = G
    return G


def step_priority(run: EventRun, G: Any) -> Any:
    """Exposure per area and ranked priorities (GeoJSON + CSV)."""
    import geopandas as gpd

    from src.priority.exposure import exposure_by_area
    from src.priority.rank import rank_areas
    from src.routing.route import load_vehicles

    cfg = load_config(repo_root() / "configs" / "priority.yaml")
    areas = gpd.read_file(data_path(cfg["admin"]["path"]))
    from shapely.geometry import box

    areas = areas[areas.intersects(box(*cfg["analysis_bbox"]))]
    flood = run.outputs.get("mask") or run.outputs["prob"]
    exposure = exposure_by_area(
        flood,
        data_path(cfg["population"]["path"]),
        areas,
        cfg["admin"]["name_field"],
        cfg["threshold"],
        cfg["method"],
    )
    facilities = gpd.read_file(data_path(cfg["facilities"]["path"]))
    points = [(g.centroid.x, g.centroid.y) for g in facilities.geometry]
    ranked = rank_areas(
        exposure,
        G,
        points,
        load_vehicles(),
        run.outputs.get("std"),
        cfg["vehicle"],
        cfg["verify_below"],
    )
    ranked.to_file(run.out_dir / "priorities.geojson", driver="GeoJSON")
    ranked.drop(columns="geometry").to_csv(run.out_dir / "priorities.csv", index=False)
    run.outputs["priorities"] = run.out_dir / "priorities.geojson"
    return ranked


def step_alert(run: EventRun, ranked: Any) -> Any:
    """CAP draft for the top-priority area (draft only)."""
    from src.alerts.approval import AlertRecord
    from src.alerts.cap import AlertContent, build_alert, polygon_from_geometry, sms_text

    cfg = load_config(repo_root() / "configs" / "priority.yaml")["alert"]
    top = ranked.iloc[0]
    depth = (
        float(np.nanmax(_raster_max(run.outputs.get("depth")))) if run.outputs.get("depth") else 1.0
    )
    content = AlertContent(
        river=cfg["river"],
        district=str(top["area_name"]),
        shelter=cfg["shelter"],
        depth_m=min(depth, 5.0),
        area_desc=f"{top['area_name']} (priority 1); {cfg['area_note']}",
        polygon=polygon_from_geometry(top.geometry),
    )
    xml = build_alert(content)
    record = AlertRecord(alert_id=f"{run.scene.stem}-p1", cap_xml=xml, sms=sms_text(content))
    (run.out_dir / f"{record.alert_id}.xml").write_text(xml, encoding="utf-8")
    run.outputs["alert"] = record
    return record


def _raster_max(path: Path | None) -> float:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
    return float(np.ma.max(arr)) if arr.count() else 1.0


def run_event(
    scene: Path,
    out_dir: Path,
    model_config: str = "configs/E6ens.yaml",
    hand: Path | None = None,
    road_status: Path | None = None,
) -> EventRun:
    """All steps in order; returns paths, results and minutes per step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run = EventRun(scene=scene, out_dir=out_dir)
    run.timed("level1", lambda: step_level1(run, model_config, hand))
    run.timed("mask", lambda: step_mask(run))
    run.timed("depth", lambda: step_depth(run))
    run.timed("level2", lambda: step_level2(run, road_status))
    G = run.timed("roads", lambda: step_roads(run))
    ranked = run.timed("priority", lambda: step_priority(run, G))
    run.timed("alert", lambda: step_alert(run, ranked))
    log.info("image to alert draft: %.2f min %s", run.total_minutes, run.minutes)
    return run
