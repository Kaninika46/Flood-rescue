"""One command for a whole event (prompt D2).

    python scripts/run_event.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif
    python scripts/run_event.py --scene ... --maxar-tiles $DATA_ROOT/nepal/maxar/tiles

Level 1 predict -> depth -> (Level 2 road status if Maxar tiles were evaluated) -> road status
and routes -> priority -> alert draft, stored in the database (``DATABASE_URL``) with the time of
each step. With ``--maxar-tiles``, run the Level 2 evaluation first in the sn8 env:
``python -m src.level2_roads.evaluate --config configs/S4.yaml --split nepal_maxar_labelled``;
its ``road_status.geojson`` is picked up here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger

log = get_logger("run_event")
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    scene: Path = typer.Option(..., help="Sentinel-1 VV/VH GeoTIFF in dB."),
    maxar_tiles: Path = typer.Option(None, help="Folder of Maxar tile pairs (Level 2)."),
    model_config: str = typer.Option("configs/E6ens.yaml"),
    hand: Path = typer.Option(None, help="HAND raster on the scene grid (built if missing)."),
    no_db: bool = typer.Option(False, help="Do not store in the database."),
) -> None:
    """Run the pipeline and store everything."""
    from src.event_pipeline import run_event

    road_status = None
    if maxar_tiles is not None:
        s4 = load_config(repo_root() / "configs" / "S4.yaml")
        road_status = data_path(
            "eval", s4["experiment"], "nepal_maxar_labelled", "road_status.geojson"
        )
        if not road_status.exists():
            log.warning("no Level 2 road status at %s: run the sn8 evaluation first", road_status)
    out = data_path("events", scene.stem)
    run = run_event(scene, out, model_config, hand, road_status)
    if not no_db:
        from sqlalchemy.orm import sessionmaker

        from backend.app.db import get_engine, init_db
        from backend.app.models import Scene
        from backend.app.services import store_event

        init_db()
        with sessionmaker(bind=get_engine(), expire_on_commit=False)() as session:
            row = Scene(name=scene.stem, s1_path=str(scene), state="processing")
            session.add(row)
            session.commit()
            try:
                store_event(session, row, run)
            except Exception:
                session.rollback()
                row.state = "failed"  # never left "processing" after an error
                session.commit()
                raise
            log.info("stored scene %d", row.id)
    for step, minutes in run.minutes.items():
        print(f"{step:10s} {minutes:7.2f} min")
    print(f"{'total':10s} {run.total_minutes:7.2f} min (image to alert draft)")


if __name__ == "__main__":
    app()
