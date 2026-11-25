"""Score a Level 2 evaluation folder (prompt B3). Run in the "sn8" environment.

    python scripts/score_sn8.py --config configs/S0.yaml --split sn8_val

Reads ``$DATA_ROOT/eval/<experiment>/<split>/`` (written by ``src.level2_roads.evaluate``) and
reports:

* **APLS** (length-weighted) with the SpaceNet 8 APLS code (``src/level2_roads/apls``, an
  adapted copy of ``third_party/spacenet8/03-sianalytics/code/apls``; it is copied into the
  evaluation folder at run time because it writes its outputs next to itself), one run per
  area because tile ids repeat across areas;
* pixel IoU / precision / recall: buildings, roads, flooded buildings, flooded roads;
* flooded-road precision / recall **by length** and by segment (``src/level2_roads/scoring.py``),
  predicted roads from the submission CSV vs the annotated roads.

Writes ``docs/results/level2/<experiment>__<split>.json`` and logs an ``eval`` run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger, git_info, start_run
from src.common.metrics import SplitScorer
from src.level2_roads.data import flood_class_map, read_image, read_sn8_split
from src.level2_roads.scoring import (
    is_flooded,
    length_precision_recall,
    lon_lat_order,
    tile_bounds,
)

log = get_logger("score_sn8")
app = typer.Typer(add_completion=False, help=__doc__)
APLS_SRC = repo_root() / "src" / "level2_roads" / "apls"  # adapted copy, see its headers


def pixel_scores(rows: pd.DataFrame, preds: Path) -> dict[str, SplitScorer]:
    """Pooled pixel scores over the tiles that have predictions."""
    scorers = {k: SplitScorer() for k in ("building", "road", "flooded_building", "flooded_road")}
    for row in rows.to_dict("records"):
        stem = Path(row["preimg"]).stem
        paths = {k: preds / k / f"{stem}_{k}pred.tif" for k in ("building", "roadspeed", "flood")}
        if not all(p.exists() for p in paths.values()):
            continue
        event = row["aoi"]
        building = read_image(paths["building"])[0] > 0
        road = read_image(paths["roadspeed"])[-1] >= 128
        flood = read_image(paths["flood"])[0]
        gt_flood = flood_class_map(read_image(data_path(row["flood"])))
        scorers["building"].update(
            building, read_image(data_path(row["building"]))[0].astype(np.int16), event=event
        )
        scorers["road"].update(
            road, read_image(data_path(row["road"]))[0].astype(np.int16), event=event
        )
        scorers["flooded_building"].update(
            flood == 2, (gt_flood == 2).astype(np.int16), event=event
        )
        scorers["flooded_road"].update(flood == 4, (gt_flood == 4).astype(np.int16), event=event)
    return scorers


def flooded_road_vectors(rows: pd.DataFrame, submission: Path) -> dict[str, float]:
    """Length-based flooded-road precision / recall over the split's tiles."""
    import geopandas as gpd
    import shapely.wkt

    truth = []
    for row in rows.to_dict("records"):
        gdf = gpd.read_file(data_path(row["label"]))
        if gdf.empty:
            continue
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        truth += [g for g, f in zip(lines.geometry, lines["flooded"], strict=True) if is_flooded(f)]
    sub = pd.read_csv(submission)
    sub.columns = [c.lower() for c in sub.columns]  # roads use WKT_Geo, buildings Wkt_Geo
    tiles = {Path(p).stem: data_path(p) for p in rows["preimg"]}
    sub = sub[(sub["object"] == "Road") & sub["imageid"].isin(set(tiles))]
    pred = []
    for image_id, wkt, flag in zip(sub["imageid"], sub["wkt_geo"], sub["flooded"], strict=True):
        if is_flooded(flag) and isinstance(wkt, str) and "EMPTY" not in wkt:
            pred.append(lon_lat_order(shapely.wkt.loads(wkt), tile_bounds(tiles[image_id])))
    if not truth and not pred:
        return length_precision_recall([], [])
    frame = gpd.GeoSeries(truth + pred, crs="EPSG:4326")
    utm = frame.estimate_utm_crs()
    projected = list(frame.to_crs(utm))
    return length_precision_recall(projected[len(truth) :], projected[: len(truth)])


def run_apls(rows: pd.DataFrame, out: Path) -> dict[str, Any]:
    """APLS per area on the split's tiles; returns the mean APLS per area and overall."""
    work = out / "apls"
    code = work / "code" / "apls"
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(APLS_SRC, code)
    results: dict[str, Any] = {}
    all_scores = []
    for aoi, group in rows.groupby("aoi"):
        truth_dir, im_dir = work / aoi / "truth", work / aoi / "images"
        prop_dir = work / aoi / "props"
        for d in (truth_dir, im_dir, prop_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        graphs = out / "roads" / "graphs_nospeed"
        for row in group.to_dict("records"):
            stem = Path(row["preimg"]).stem
            (truth_dir / Path(row["label"]).name).symlink_to(data_path(row["label"]))
            (im_dir / Path(row["preimg"]).name).symlink_to(data_path(row["preimg"]))
            graph = graphs / f"{stem}_roadspeedpred.gpickle"
            if graph.exists():  # APLS pairs files by "*<tile id>.gpickle"
                (prop_dir / f"{stem}.gpickle").symlink_to(graph)
        name = f"apls_{aoi}"
        cmd = [
            sys.executable,
            str(code / "apls_sn8.py"),
            "--test_method",
            "gt_json_prop_pkl",
            "--output_name",
            name,
            "--truth_dir",
            str(truth_dir) + "/",
            "--prop_dir",
            str(prop_dir),
            "--im_dir",
            str(im_dir) + "/",
            "--max_files",
            "100000",
            "--n_plots",
            "0",
        ]
        log.info("APLS for %s (%d tiles)", aoi, len(group))
        subprocess.run(cmd, check=True, cwd=code)
        csvs = sorted((work / "code" / "outputs" / name).rglob("scores__*.csv"))
        if not csvs:
            raise FileNotFoundError(f"APLS wrote no scores for {aoi}")
        df = pd.read_csv(csvs[-1])
        per_tile = df[~df.iloc[:, 1].astype(str).isin(["means", "stds"])]
        scores = pd.to_numeric(per_tile["APLS"], errors="coerce").dropna().tolist()
        results[aoi] = float(np.mean(scores)) if scores else float("nan")
        all_scores += scores
    results["overall"] = float(np.mean(all_scores)) if all_scores else float("nan")
    return results


@app.command()
def main(
    config: Path = typer.Option(...),
    split: str = typer.Option(...),
    skip_apls: bool = typer.Option(False, help="Pixel and vector scores only."),
) -> None:
    """Compute and save all Level 2 scores for one evaluation folder."""
    cfg = load_config(config)
    out = data_path("eval", cfg["experiment"], split)
    rows = read_sn8_split(split)
    rows = rows[[data_path(p).exists() for p in rows["preimg"]]]
    rows = rows[
        [
            (out / "preds" / "flood" / f"{Path(p).stem}_floodpred.tif").exists()
            for p in rows["preimg"]
        ]
    ]
    log.info("%d tiles with predictions", len(rows))
    pixels = pixel_scores(rows, out / "preds")
    result: dict[str, Any] = {
        "experiment": cfg["experiment"],
        "split": split,
        "tiles": len(rows),
        "git_commit": git_info()["git_commit"],
        "pixel": {k: v.summary() for k, v in pixels.items()},
    }
    submission = out / "roads_submission.csv"
    if submission.exists():
        result["flooded_road_by_length"] = flooded_road_vectors(rows, submission)
    status = out / "road_status_summary.json"
    if status.exists():  # S3: how many segments are unknown (cloud) or need verification
        result["road_status_counts"] = json.loads(status.read_text(encoding="utf-8"))
    if not skip_apls and (out / "roads" / "graphs_nospeed").is_dir():
        result["apls"] = run_apls(rows, out)
    flat = {f"{split}/{k}/iou": v.overall.iou for k, v in pixels.items()}
    flat[f"{split}/flooded_road/pixel_recall"] = pixels["flooded_road"].overall.recall
    for k, v in result.get("flooded_road_by_length", {}).items():
        flat[f"{split}/flooded_road_len/{k}"] = v
    if "apls" in result:
        flat[f"{split}/apls"] = result["apls"]["overall"]
    for k, v in result.get("road_status_counts", {}).items():
        flat[f"{split}/segments_{k}"] = v
    with start_run(cfg, cfg.get("seeds", [0])[0], job_type="eval") as run:
        run.log_summary(flat)
    path = repo_root() / "docs" / "results" / "level2" / f"{cfg['experiment']}__{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log.info("wrote %s", path)
    for k, v in flat.items():
        print(f"  {k:45s} {v:.4f}" if isinstance(v, float) else f"  {k:45s} {v}")


if __name__ == "__main__":
    app()
