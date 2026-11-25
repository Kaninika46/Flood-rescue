"""S4: the best Level 2 model on the labelled Nepal 2024 Maxar tiles (prompt B8).

    python scripts/nepal_level2.py prepare     # sn8 env: labels -> SpaceNet 8 masks + split file
    python -m src.level2_roads.evaluate --config configs/S4.yaml --split nepal_maxar_labelled
    python scripts/nepal_level2.py compare     # flood env: Level 2 vs radar+depth on the same roads

You label the GeoJSONs in ``tiles/labels/`` (written by ``maxar_nepal.py osm``).
``prepare`` writes cleaned copies to ``tiles/annotations/``, the mapping CSV the baseline scripts
expect, runs our copies of
``geojson_prep.py`` and ``create_masks.py`` on ``$DATA_ROOT/nepal/maxar/tiles`` and writes
``splits/nepal_maxar_labelled.csv``. Segments labelled ``unclear`` are kept out of the masks'
flood channels by mapping them to ``null`` (the baseline then treats them as not flooded), and
are counted separately by ``compare``.

``compare`` takes every labelled road segment (``yes`` / ``no``) and decides "flooded" three ways:
Level 2 (predicted flooded roads within 5 m covering half the segment), Level 1 radar (flood
probability >= 0.5 on at least 20 % of the samples along it) and Level 1 depth (any sample deeper
than 0.15 m, the car limit). Writes precision / recall of each, plus agreement, to
``docs/results/nepal_level2.json`` and ``docs/nepal_level2.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger
from src.level2_roads.data import MASK_KINDS

log = get_logger("nepal_level2")
app = typer.Typer(add_completion=False, help=__doc__)
TILES = "nepal/maxar/tiles"


def label_value(value: Any) -> str | None:
    """Normalise a ``flooded`` label to yes / no / unclear / None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().lower()
    return {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "unclear": "unclear",
    }.get(text)


@app.command()
def prepare() -> None:
    """Mapping CSV, baseline data prep and the split file for the labelled tiles."""
    import geopandas as gpd

    root = data_path(TILES)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    rows, mapping = [], []
    for geo in sorted((root / "labels").glob("*.geojson")):  # your QGIS labels
        tile_id = geo.stem
        pre = sorted((root / "PRE-event").glob(f"*_{tile_id}.tif"))
        post = sorted((root / "POST-event").glob(f"*_{tile_id}.tif"))
        if not pre or not post:
            continue
        gdf = gpd.read_file(geo)
        labels = gdf["flooded"].map(label_value) if "flooded" in gdf else pd.Series(dtype=object)
        if labels.isna().all():
            continue  # not labelled yet
        clean = gdf.copy()
        clean["flooded"] = ["yes" if v == "yes" else None for v in labels]
        clean.to_file(root / "annotations" / geo.name, driver="GeoJSON")  # for the baseline
        mapping.append(
            {
                "label": geo.name,
                "pre-event image": pre[0].name,
                "post-event image 1": post[0].name,
                "post-event image 2": "",
            }
        )
        stem = pre[0].stem
        row = {
            "aoi": "tiles",
            "event": "Nepal",
            "tile_id": tile_id,
            "preimg": f"{TILES}/PRE-event/{pre[0].name}",
            "postimg": f"{TILES}/POST-event/{post[0].name}",
            "postimg2": "",
            "label": f"{TILES}/annotations/{geo.name}",
        }
        row.update({k: f"{TILES}/annotations/masks/{k}_{stem}.tif" for k in MASK_KINDS})
        rows.append(row)
    if not rows:
        raise typer.Exit("no labelled tiles found (see docs/labelling_guide.md)")
    pd.DataFrame(mapping).to_csv(root / "tiles_label_image_mapping.csv", index=False)
    prep = repo_root() / "src" / "level2_roads" / "sn8" / "data_prep"
    for script in ("geojson_prep.py", "create_masks.py"):
        subprocess.run(
            [
                sys.executable,
                str(prep / script),
                "--root_dir",
                str(root.parent),
                "--aoi_dirs",
                root.name,
            ],
            check=True,
            cwd=prep,
        )
    out = repo_root() / "splits" / "nepal_maxar_labelled.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("%d labelled tiles -> %s", len(rows), out)


def sample_line(raster: Path, line: Any, line_crs: Any, step_m: float = 5.0) -> list[float]:
    """Raster values every ``step_m`` metres along a line (NaN outside the raster)."""
    import geopandas as gpd
    import numpy as np
    import rasterio

    with rasterio.open(raster) as src:
        geom = gpd.GeoSeries([line], crs=line_crs).to_crs(src.crs).iloc[0]
        metric = (
            gpd.GeoSeries([line], crs=line_crs)
            .to_crs(gpd.GeoSeries([line], crs=line_crs).estimate_utm_crs())
            .iloc[0]
        )
        n = max(2, int(metric.length / step_m) + 1)
        points = [geom.interpolate(f, normalized=True) for f in np.linspace(0, 1, n)]
        values = [v[0] for v in src.sample([(p.x, p.y) for p in points], masked=True)]
        return [float(v) if v is not np.ma.masked else float("nan") for v in values]


@app.command()
def compare(buffer_m: float = typer.Option(5.0), car_limit_m: float = typer.Option(0.15)) -> None:
    """Level 2 vs Level 1 (radar, depth) on the labelled road segments."""
    import geopandas as gpd
    import numpy as np
    import shapely.wkt

    from src.common.metrics import Counts
    from src.level2_roads.scoring import is_flooded

    cfg = load_config(repo_root() / "configs" / "S4.yaml")
    nepal = load_config(repo_root() / "configs" / "nepal.yaml")
    split = pd.read_csv(repo_root() / "splits" / "nepal_maxar_labelled.csv")
    sub_path = data_path("eval", cfg["experiment"], "nepal_maxar_labelled", "roads_submission.csv")
    sub = pd.read_csv(sub_path)
    sub.columns = [c.lower() for c in sub.columns]
    level1 = data_path(nepal["paths"]["level1_dir"])
    probs = sorted(level1.glob("*_flood_prob.tif"))
    depths = sorted(level1.glob("*_depth.tif"))
    segments = []
    for row in split.to_dict("records"):
        gdf = gpd.read_file(data_path(TILES, "labels", Path(row["label"]).name))  # with unclear
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        stem = Path(row["preimg"]).stem
        pred = sub[(sub["object"] == "Road") & (sub["imageid"] == stem)]
        flooded_pred = [
            shapely.wkt.loads(w)
            for w, f in zip(pred["wkt_geo"], pred["flooded"], strict=True)
            if is_flooded(f) and isinstance(w, str) and "EMPTY" not in w
        ]
        utm = gdf.estimate_utm_crs()
        zone = (
            gpd.GeoSeries(flooded_pred, crs="EPSG:4326").to_crs(utm).buffer(buffer_m).union_all()
            if flooded_pred
            else None
        )
        for geom, label in zip(gdf.geometry, gdf["flooded"].map(label_value), strict=True):
            if label not in ("yes", "no"):
                continue
            g_utm = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(utm).iloc[0]
            l2 = bool(zone is not None and g_utm.intersection(zone).length >= 0.5 * g_utm.length)
            seg = {"truth": label == "yes", "level2": l2, "radar": None, "depth": None}
            if probs:
                values = np.array(sample_line(probs[0], geom, gdf.crs))
                if np.isfinite(values).any():
                    seg["radar"] = bool(np.nanmean(values >= 0.5) >= 0.2)
            if depths:
                values = np.array(sample_line(depths[0], geom, gdf.crs))
                if np.isfinite(values).any():
                    seg["depth"] = bool(np.nanmax(np.nan_to_num(values)) > car_limit_m)
            segments.append(seg)
    df = pd.DataFrame(segments)
    result: dict[str, Any] = {"segments": len(df), "truly_flooded": int(df["truth"].sum())}
    for method in ("level2", "radar", "depth"):
        sub_df = df[df[method].notna()]
        c = Counts(
            tp=int((sub_df[method].astype(bool) & sub_df["truth"]).sum()),
            fp=int((sub_df[method].astype(bool) & ~sub_df["truth"]).sum()),
            fn=int((~sub_df[method].astype(bool) & sub_df["truth"]).sum()),
        )
        result[method] = {
            "segments_covered": len(sub_df),
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
        }
    both = df[df["radar"].notna()]
    result["agreement_level2_radar"] = (
        float((both["level2"] == both["radar"].astype(bool)).mean()) if len(both) else None
    )
    path = repo_root() / "docs" / "results" / "nepal_level2.json"
    path.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    lines = [
        "# Nepal 2024, Level 2 (S4) and comparison with Level 1",
        "",
        "Generated by `python scripts/nepal_level2.py compare`. Segments labelled `unclear` "
        "are excluded (docs/labelling_guide.md).",
        "",
        f"Labelled road segments: {result['segments']} ({result['truly_flooded']} flooded).",
        "",
        "| Status from | Segments covered | Precision | Recall (key safety number) | F1 |",
        "|---|---|---|---|---|",
    ]
    for method, name in (
        ("level2", "Level 2 (S4 model)"),
        ("radar", "Level 1 radar probability"),
        ("depth", f"Level 1 depth > {car_limit_m} m"),
    ):
        r = result[method]
        lines.append(
            f"| {name} | {r['segments_covered']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} |"
        )
    (repo_root() / "docs" / "nepal_level2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s and docs/nepal_level2.md", path)


@app.callback()
def _group() -> None:
    """Nepal Level 2 test (see the module docstring)."""


if __name__ == "__main__":
    app()
