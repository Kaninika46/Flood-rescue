"""Draw safe routes on the Nepal replay (prompt B9 "done when"). Run in the "flood" env.

    python scripts/route_demo.py graph        # download the OSM drive network once (Overpass)
    python scripts/route_demo.py run          # statuses from Level 1 (+ Level 2), routes, PNG

Writes ``docs/figures/nepal_routes.png`` (roads coloured by status, one route per vehicle,
"verify" segments circled) and ``$DATA_ROOT/nepal/routes/<vehicle>.geojson``, and prints each
route's time, number of verify segments and unsafe length (target 0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger
from src.routing.graph import download_graph, load_graph, save_graph
from src.routing.route import NoRoute, load_vehicles, route, unsafe_length_m
from src.routing.status import StatusSettings, assign_status, status_counts

log = get_logger("route_demo")
app = typer.Typer(add_completion=False, help=__doc__)
COLOURS = {
    "dry": "#9e9e9e",
    "flooded": "#1565c0",
    "verify": "#f9a825",
    "unknown": "#8e24aa",
    "closed": "#c62828",
}


def _cfg() -> dict:
    return load_config(repo_root() / "configs" / "routing.yaml")


@app.command()
def graph() -> None:
    """Download and save the road graph for the configured bbox."""
    cfg = _cfg()["graph"]
    G = download_graph(tuple(cfg["bbox"]), cfg["network_type"])
    path = save_graph(G, data_path(cfg["path"]))
    log.info("%d nodes, %d edges -> %s", len(G), G.number_of_edges(), path)


@app.command()
def run() -> None:
    """Assign statuses, route every demo vehicle and draw the map."""
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import shape

    from src.routing.graph import edge_geometry

    cfg = _cfg()
    G = load_graph(data_path(cfg["graph"]["path"]))
    inputs = {k: data_path(v) for k, v in cfg["inputs"].items()}
    level2 = gpd.read_file(inputs["level2_status"]) if inputs["level2_status"].exists() else None
    assign_status(
        G,
        depth=inputs["depth"] if inputs["depth"].exists() else None,
        std=inputs["std"] if inputs["std"].exists() else None,
        level2=level2,
        settings=StatusSettings(**cfg["status"]),
    )
    log.info("edge statuses: %s", dict(status_counts(G)))
    profiles = load_vehicles()
    fig, ax = plt.subplots(figsize=(10, 10))
    for u, v, data in G.edges(data=True):
        xs, ys = edge_geometry(G, u, v, data).xy
        ax.plot(xs, ys, color=COLOURS.get(data.get("status"), "#bdbdbd"), lw=0.6, zorder=1)
    out_dir = data_path("nepal", "routes")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, vehicle in enumerate(cfg["demo"]["vehicles"]):
        try:
            res = route(
                G, tuple(cfg["demo"]["start"]), tuple(cfg["demo"]["end"]), vehicle, profiles
            )
        except NoRoute:
            log.warning("%s: no passable route", vehicle)
            continue
        (out_dir / f"{vehicle}.geojson").write_text(json.dumps(res), encoding="utf-8")
        line = shape(res["features"][0]["geometry"])
        ids = res["features"][0]["properties"]["edge_ids"]
        unsafe = unsafe_length_m(G, ids, vehicle, profiles)
        ax.plot(
            *line.xy,
            lw=3 - i * 0.6,
            zorder=3,
            label=f"{vehicle}: {res['total_time_min']} min, "
            f"{len(res['verify_segments'])} verify, unsafe {unsafe:.0f} m",
        )
        for f in res["features"][1:]:
            seg = shape(f["geometry"])
            ax.scatter(*seg.centroid.xy, s=60, facecolors="none", edgecolors="#f9a825", zorder=4)
        log.info(
            "%s: %.1f min, %d verify segments, unsafe length %.0f m",
            vehicle,
            res["total_time_min"],
            len(res["verify_segments"]),
            unsafe,
        )
    ax.set_title(
        "Safe routes on the Koshi replay (advice; the team on the ground decides)\n"
        "grey dry, blue flooded, yellow verify, purple unknown, red closed",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    out = repo_root() / "docs" / "figures" / "nepal_routes.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    log.info("wrote %s", out)


if __name__ == "__main__":
    app()
