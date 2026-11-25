"""SYNTHETIC demo event for the dashboard and the app, before the real Nepal data exists.

    export DATA_ROOT=~/flood-demo-data          # a NEW, empty folder (never your real data)
    python scripts/demo_seed.py

Writes made-up inputs into ``$DATA_ROOT`` at the paths the configs expect (a 12 x 12 street grid
near Biratnagar, a winding "river" flood, population, 16 wards and two facilities), then runs the
real road-status, priority and alert-draft steps of ``src/event_pipeline.py`` on them and stores the
result in the database (``DATABASE_URL``) as a scene named "SYNTHETIC demo". The Level 1 and depth
steps are replaced by the synthetic rasters. Nothing here is a real flood map: the scene name,
the ward names and the alert text all say so.

The script refuses to write into a ``$DATA_ROOT`` that is not empty, unless the folder was made
by this script (it leaves a ``DEMO_SYNTHETIC.txt`` marker).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
import typer
from rasterio.transform import from_origin

from src.common.config import data_path, get_data_root, load_config, repo_root
from src.common.logging import get_logger
from src.common.raster import write_raster

log = get_logger("demo_seed")
app = typer.Typer(add_completion=False, help=__doc__)

MARKER = "DEMO_SYNTHETIC.txt"
SCENE = "SYNTHETIC demo (not real data)"
WEST, SOUTH, EAST, NORTH = 87.13, 26.44, 87.30, 26.61
PIXEL_DEG = 0.0003  # about 30 m
GRID_N = 12


def river_lon(lat: np.ndarray) -> np.ndarray:
    """Centre line of the made-up river (winds north to south)."""
    return 87.215 + 0.018 * np.sin((lat - SOUTH) * 38.0)


def synthetic_rasters(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Flood probability, ensemble std and water depth (m) on a ~30 m lon/lat grid."""
    w = round((EAST - WEST) / PIXEL_DEG)
    h = round((NORTH - SOUTH) / PIXEL_DEG)
    lons = WEST + (np.arange(w) + 0.5) * PIXEL_DEG
    lats = NORTH - (np.arange(h) + 0.5) * PIXEL_DEG
    lon, lat = np.meshgrid(lons, lats)
    dist_km = np.abs(lon - river_lon(lat)) * 100.0  # ~100 km per degree of longitude here
    width_km = 1.2 + 0.8 * np.sin(lat * 90.0) ** 2  # the flood widens and narrows
    north_fade = np.clip((26.575 - lat) / 0.015, 0, 1)  # the far north stays dry: one
    # bridge is outside the flood, so routes exist and a field report can change them
    prob = np.exp(-0.5 * (dist_km / width_km) ** 4) * north_fade
    prob = prob + rng.normal(0, 0.03, lon.shape)
    prob = np.clip(prob, 0, 1).astype(np.float32)
    std = (0.3 * 4 * prob * (1 - prob)).astype(np.float32)  # unsure at the flood edge
    depth = np.where(prob > 0.5, 2.6 * (prob - 0.5) * 2 + 0.05, np.nan).astype(np.float32)
    return prob, std, depth, from_origin(WEST, NORTH, PIXEL_DEG, PIXEL_DEG)


def synthetic_graph() -> nx.MultiDiGraph:
    """Two-way street grid in OSMnx format; streets that cross the river are bridges."""
    G = nx.MultiDiGraph(crs="EPSG:4326")
    xs = np.linspace(WEST + 0.005, EAST - 0.005, GRID_N)
    ys = np.linspace(SOUTH + 0.005, NORTH - 0.005, GRID_N)
    node = {}
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            nid = 1000 + i * GRID_N + j
            node[i, j] = nid
            G.add_node(nid, x=float(x), y=float(y), osmid=nid)
    osmid = 50_000
    for i in range(GRID_N):
        for j in range(GRID_N):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= GRID_N or j + dj >= GRID_N:
                    continue
                a, b = node[i, j], node[i + di, j + dj]
                ax, ay, bx, by = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[b]["x"], G.nodes[b]["y"]
                mid_lat = (ay + by) / 2
                crosses = dj and min(ax, bx) < river_lon(np.array(mid_lat)) < max(ax, bx)
                main = i % 4 == 0 or j % 4 == 0
                length = float(np.hypot((bx - ax) * 99_700, (by - ay) * 110_900))
                speed = 50.0 if main else 30.0
                attrs = {
                    "osmid": osmid,
                    "highway": "primary" if main else "residential",
                    "name": f"Synthetic {'road' if main else 'lane'} {osmid}",
                    "oneway": False,
                    "length": round(length, 1),
                    "speed_kph": speed,
                    "travel_time": round(length / (speed / 3.6), 1),
                }
                if crosses:
                    attrs["bridge"] = "yes"
                osmid += 1
                G.add_edge(a, b, key=0, reversed=False, **attrs)
                G.add_edge(b, a, key=0, reversed=True, **attrs)
    return G


def synthetic_population(rng: np.random.Generator) -> tuple[np.ndarray, object]:
    """People per ~100 m cell: rural background plus three made-up towns."""
    step = 0.001
    w, h = int((EAST - WEST) / step), int((NORTH - SOUTH) / step)
    lon, lat = np.meshgrid(WEST + (np.arange(w) + 0.5) * step, NORTH - (np.arange(h) + 0.5) * step)
    pop = rng.gamma(2.0, 3.0, lon.shape)
    for tx, ty, peak, r in (
        (87.27, 26.46, 180, 0.012),
        (87.16, 26.59, 90, 0.008),
        (87.21, 26.53, 120, 0.010),
    ):
        pop += peak * np.exp(-((lon - tx) ** 2 + (lat - ty) ** 2) / (2 * r**2))
    return pop.astype(np.float32), from_origin(WEST, NORTH, step, step)


def synthetic_areas() -> object:
    """16 made-up wards on a 4 x 4 grid."""
    import geopandas as gpd
    from shapely.geometry import box

    xs = np.linspace(WEST, EAST, 5)
    ys = np.linspace(SOUTH, NORTH, 5)
    rows = []
    for i in range(4):
        for j in range(4):
            rows.append(
                {
                    "shapeName": f"Synthetic ward {i * 4 + j + 1}",
                    "geometry": box(xs[j], ys[i], xs[j + 1], ys[i + 1]),
                }
            )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def check_root(root: Path) -> None:
    """Refuse a non-empty DATA_ROOT that this script did not create."""
    if (root / MARKER).exists():
        return
    if root.exists() and any(root.iterdir()):
        raise typer.BadParameter(
            f"{root} is not empty and has no {MARKER}: point DATA_ROOT at a new folder so the "
            "synthetic files never mix with real data."
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text(
        "Everything in this folder is SYNTHETIC demo data "
        "(scripts/demo_seed.py). Do not use it for results.\n"
    )


@app.command()
def main(seed: int = 0, no_db: bool = typer.Option(False, help="Only write the files.")) -> None:
    """Write the synthetic inputs, run roads -> priority -> alert, store the scene."""
    import geopandas as gpd
    from shapely.geometry import Point

    from src.event_pipeline import EventRun, step_alert, step_priority, step_roads
    from src.routing.graph import save_graph

    root = get_data_root(must_exist=False)
    check_root(root)
    rng = np.random.default_rng(seed)
    routing = load_config(repo_root() / "configs" / "routing.yaml")
    priority = load_config(repo_root() / "configs" / "priority.yaml")

    out = data_path("events", "SYNTHETIC_demo")
    prob, std, depth, transform = synthetic_rasters(rng)
    paths = {}
    for name, arr in (("prob", prob), ("std", std), ("depth", depth)):
        paths[name] = write_raster(
            out / f"SYNTHETIC_demo_{name}.tif",
            arr,
            crs="EPSG:4326",
            transform=transform,
            nodata=float("nan"),
        )
    save_graph(synthetic_graph(), data_path(routing["graph"]["path"]))
    pop, pop_transform = synthetic_population(rng)
    write_raster(
        data_path(priority["population"]["path"]),
        pop,
        crs="EPSG:4326",
        transform=pop_transform,
        nodata=-99999.0,
    )
    areas_path = data_path(priority["admin"]["path"])
    areas_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_areas().to_file(areas_path, driver="GeoJSON")
    fac_path = data_path(priority["facilities"]["path"])
    fac_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["Synthetic hospital", "Synthetic health post"]},
        geometry=[Point(87.272, 26.455), Point(87.155, 26.595)],
        crs="EPSG:4326",
    ).to_file(fac_path, driver="GeoJSON")
    log.info("synthetic inputs written under %s", root)

    run = EventRun(scene=Path("SYNTHETIC_demo"), out_dir=out)
    run.outputs.update(paths)
    run.minutes.update(level1=0.0, depth=0.0, level2=0.0)  # replaced by synthetic rasters
    G = run.timed("roads", lambda: step_roads(run))
    ranked = run.timed("priority", lambda: step_priority(run, G))
    run.timed("alert", lambda: step_alert(run, ranked))
    log.info("edge statuses: %s", run.outputs["edge_status_counts"])
    for vehicle, r in run.outputs["routes"].items():
        log.info(
            "demo route %s: %s",
            vehicle,
            "no safe route"
            if r is None
            else f"{r['total_time_min']:.1f} min, {len(r['verify_segments'])} verify segments",
        )
    if no_db:
        return

    from sqlalchemy.orm import sessionmaker

    from backend.app.db import get_engine, init_db
    from backend.app.models import Scene
    from backend.app.services import store_event

    init_db()
    with sessionmaker(bind=get_engine(), expire_on_commit=False)() as session:
        scene = Scene(name=SCENE, s1_path="(synthetic)", state="processing")
        session.add(scene)
        session.commit()
        store_event(session, scene, run)
        log.info("stored scene %d (%s)", scene.id, SCENE)


if __name__ == "__main__":
    app()
