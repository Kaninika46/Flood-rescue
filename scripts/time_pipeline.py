"""Minutes from image file to alert draft on one Nepal scene (prompt C5).

    python scripts/time_pipeline.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif

Runs ``src/event_pipeline.run_event`` (Level 1 predict, depth, Level 2 status if present, road
status + routes, priority, alert draft) and writes the minutes per step and in total, with the GPU
name, to ``docs/system_metrics.md`` and ``docs/results/system_metrics.json``. It also reports the
route safety metric: length of each demo route through water deeper than its vehicle limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common import gpu
from src.common.config import data_path, repo_root
from src.event_pipeline import run_event
from src.routing.route import unsafe_length_m

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    scene: Path = typer.Option(..., help="Sentinel-1 VV/VH GeoTIFF (dB)."),
    model_config: str = typer.Option("configs/E6ens.yaml"),
    hand: Path = typer.Option(None, help="HAND on the scene grid (built if missing)."),
    road_status: Path = typer.Option(None, help="Level 2 road_status.geojson, if any."),
) -> None:
    """Run the pipeline once and write the timing report."""
    out = data_path("nepal", "event", scene.stem)
    run = run_event(scene, out, model_config, hand, road_status)
    routes = run.outputs.get("routes", {})
    route_rows = {v: (r["total_time_min"] if r else None) for v, r in routes.items()}
    G = run.outputs.get("graph")
    unsafe = {  # route safety metric: metres through water deeper than the vehicle limit
        v: unsafe_length_m(G, r["features"][0]["properties"]["edge_ids"], v)
        for v, r in routes.items()
        if r and G is not None
    }
    payload = {
        "scene": scene.name,
        "gpu": gpu.describe(),
        "minutes": run.minutes,
        "total_minutes": run.total_minutes,
        "route_minutes": route_rows,
        "unsafe_length_m": unsafe,
        "edge_status_counts": run.outputs.get("edge_status_counts"),
    }
    res = repo_root() / "docs" / "results" / "system_metrics.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    lines = [
        "# System metrics",
        "",
        f"Scene `{scene.name}`, GPU: {payload['gpu']}.",
        "",
        "| Step | Minutes |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.2f} |" for k, v in run.minutes.items()]
    lines += [
        f"| **image to alert draft** | **{run.total_minutes:.2f}** |",
        "",
        "Routes (minutes; None = no passable route): "
        + ", ".join(f"{k}: {v}" for k, v in route_rows.items()),
        "",
        "Route length through water deeper than the vehicle limit (target 0): "
        + ", ".join(f"{k}: {v:.0f} m" for k, v in unsafe.items()),
        "",
    ]
    (repo_root() / "docs" / "system_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    app()
