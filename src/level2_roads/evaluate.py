"""Level 2 evaluation in one command (prompt B3). Run in the "sn8" environment.

    python -m src.level2_roads.evaluate --config configs/S0.yaml --split sn8_val

Steps (outputs under ``$DATA_ROOT/eval/<experiment>/<split>/``):

1. ``infer``   prediction GeoTIFFs (``src/level2_roads/infer.py``)
2. ``roads``   baseline road post-processing, unchanged copies run as scripts:
               vectorize_roads -> wkt_to_G -> infer_speed -> create_submission
3. ``buildings`` baseline building_postprocessing (flooded / dry building polygons)
4. ``status``  cloud masks on the post images and per-segment road status
               (flooded / dry / unknown / verify, ``road_status.geojson``; used by routing)
5. ``score``   ``scripts/score_sn8.py``: APLS (SpaceNet 8 APLS code), pixel IoU for buildings,
               roads, flooded buildings and flooded roads, and flooded-road precision / recall by
               length; logged as an ``eval`` run and saved to ``docs/results/level2/``.

``--step`` runs a subset (e.g. ``--step score`` after a crash).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger
from src.level2_roads.data import read_sn8_split

log = get_logger("sn8_eval")
app = typer.Typer(add_completion=False, help=__doc__)
STEPS = ("infer", "roads", "buildings", "status", "score")
POST = repo_root() / "src" / "level2_roads" / "sn8" / "postprocessing"


def eval_dir(experiment: str, split: str) -> Path:
    """``$DATA_ROOT/eval/<experiment>/<split>``."""
    return data_path("eval", experiment, split)


def write_eval_csv(split: str, path: Path) -> Path:
    """The split with absolute paths (the baseline scripts expect them)."""
    df = read_sn8_split(split)
    for col in ("preimg", "postimg", "postimg2", "label", "building", "road", "roadspeed", "flood"):
        df[col] = [str(data_path(p)) if p else "" for p in df[col]]
    df = df[[Path(p).exists() for p in df["preimg"]]]
    df.to_csv(path, index=False)
    return path


def _run(cmd: list[str], cwd: Path) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def run_roads(out: Path, eval_csv: Path) -> None:
    """Road post-processing chain of ``road_post.sh`` with our folder layout."""
    roads, py = POST / "roads", sys.executable
    pred = out / "preds"
    road_dir = pred / "roadspeed"
    for sub in ("", "graphs_nospeed", "graphs_speed"):  # the scripts only make leaf folders
        (out / "roads" / sub).mkdir(parents=True, exist_ok=True)
    _run(
        [
            py,
            "vectorize_roads.py",
            "--im_dir",
            road_dir,
            "--out_dir",
            out / "roads",
            "--write_shps",
            "--write_graphs",
            "--write_csvs",
            "--write_skeletons",
        ],
        roads,
    )
    _run(
        [
            py,
            "wkt_to_G.py",
            "--wkt_submission",
            out / "roads" / "sknw_wkt.csv",
            "--graph_dir",
            out / "roads" / "graphs_nospeed",
            "--log_file",
            out / "roads" / "wkt_to_G.log",
            "--min_subgraph_length_pix",
            "20",
            "--min_spur_length_m",
            "10",
        ],
        roads,
    )
    _run(
        [
            py,
            "infer_speed.py",
            "--eval_csv",
            eval_csv,
            "--mask_dir",
            road_dir,
            "--graph_dir",
            out / "roads" / "graphs_nospeed",
            "--graph_speed_dir",
            out / "roads" / "graphs_speed",
            "--log_file",
            out / "roads" / "graph_speed.log",
        ],
        roads,
    )
    _run(
        [
            py,
            "create_submission.py",
            "--flood_pred",
            pred / "flood",
            "--graph_speed_dir",
            out / "roads" / "graphs_speed",
            "--output_csv_path",
            out / "roads_submission.csv",
            "--output_shapefile_path",
            out / "roads" / "flood_road_speed.shp",
        ],
        roads,
    )


def run_buildings(out: Path) -> None:
    """Building post-processing with the baseline's README parameters."""
    pred = out / "preds"
    (out / "buildings").mkdir(parents=True, exist_ok=True)
    # argument names of the script itself (the baseline README lists older ones)
    _run(
        [
            sys.executable,
            "building_postprocessing.py",
            "--foundation_pred_dir",
            pred / "building",
            "--flood_pred_dir",
            pred / "flood",
            "--out_submission_csv",
            out / "buildings_submission.csv",
            "--out_shapefile_dir",
            out / "buildings",
            "--square_size",
            "5",
            "--simplify_tolerance",
            "0.75",
            "--min_area",
            "5",
            "--percent_positive",
            "0.5",
        ],
        POST / "buildings",
    )


def run_status(cfg: dict, split: str, out: Path) -> None:
    """Cloud masks (post image on the pre grid) and the road status GeoJSON + summary."""
    import json

    import pandas as pd

    from src.level2_roads.cloud import cloud_mask
    from src.level2_roads.data import read_image, warp_post_to_pre, write_like_pre
    from src.level2_roads.road_status import STATUSES, segment_status

    rows = read_sn8_split(split)
    rows = rows[[data_path(p).exists() for p in rows["preimg"]]]
    cloud_dir = out / "preds" / "cloud"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for row in rows.to_dict("records"):
        pre = data_path(row["preimg"])
        _, height, width = read_image(pre).shape
        post = warp_post_to_pre(data_path(row["postimg"]), height, width)
        mask = cloud_mask(post, cfg.get("cloud"))
        write_like_pre(cloud_dir / f"{pre.stem}_cloud.tif", pre, mask.astype("uint8")[None])
        tiles[pre.stem] = pre
    submission = out / "roads_submission.csv"
    if not submission.exists():
        log.warning("no road submission; run the roads step first")
        return
    gdf = segment_status(pd.read_csv(submission), tiles, out, cfg)
    gdf.to_file(out / "road_status.geojson", driver="GeoJSON")
    counts = {s: int((gdf["status"] == s).sum()) if len(gdf) else 0 for s in STATUSES}
    (out / "road_status_summary.json").write_text(json.dumps(counts), encoding="utf-8")
    log.info("road status: %s", counts)


@app.command()
def main(
    config: Path = typer.Option(..., help="configs/S0.yaml"),
    split: str = typer.Option(..., help="e.g. sn8_val"),
    step: list[str] = typer.Option(list(STEPS), help=f"Steps: {', '.join(STEPS)}"),
    limit: int = typer.Option(0, help="Only the first N tiles (smoke tests)."),
    seed: list[int] = typer.Option([], help="Flood-network seeds (default: config seeds)."),
) -> None:
    """Run the evaluation steps for one experiment and split."""
    cfg = load_config(config)
    out = eval_dir(cfg["experiment"], split)
    out.mkdir(parents=True, exist_ok=True)
    eval_csv = write_eval_csv(split, out / "eval.csv")
    if "infer" in step:
        from src.level2_roads.infer import run_inference

        run_inference(cfg, split, out / "preds", seed or None, limit)
    if "roads" in step:
        run_roads(out, eval_csv)
    if "buildings" in step:
        run_buildings(out)
    if "status" in step:
        run_status(cfg, split, out)
    if "score" in step:
        _run(
            [
                sys.executable,
                str(repo_root() / "scripts" / "score_sn8.py"),
                "--config",
                str(config),
                "--split",
                split,
            ],
            repo_root(),
        )


if __name__ == "__main__":
    app()
