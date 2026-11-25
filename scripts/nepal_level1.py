"""Nepal radar replay for the 27 September 2024 Koshi floods (prompt A10).

Run after ``scripts/nepal_s1_download.py`` (scenes in ``$DATA_ROOT/nepal/s1``) and after the
UNOSAT flood-extent polygons were saved by hand in ``$DATA_ROOT/nepal/unosat/``:

    python scripts/nepal_level1.py run            # every step, skipping outputs that exist
    python scripts/nepal_level1.py run --step compare --overwrite

Steps (``--step`` runs one):

1. ``terrain``  HAND and slope on the post-flood scene grid (scripts/prepare_hand.py logic)
2. ``predict``  Level 1 ensemble (configs/E6ens.yaml) -> flood probability + confidence (std)
3. ``masks``    flood masks of three methods, without permanent water (JRC occurrence >= 50 %):
                ``model`` (probability >= 0.5), ``change_otsu`` (after - before VH, Nepal only,
                HAND <= 15 m) and ``otsu`` (one VH threshold for the whole post scene)
4. ``depth``    FwDET depth of the model flood mask on a 30 m UTM grid
5. ``compare``  flooded km^2 and IoU / F1 / precision / recall against UNOSAT inside the
                analysed area (UNOSAT analysis-extent polygon if provided, else the scene
                footprint), the largest disagreement patches (failure cases to inspect),
                PNG maps, ``docs/results/nepal_level1.json`` and ``docs/nepal_level1.md``

Everything runs block by block, so a full 10 m scene (about 1e9 pixels) fits in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import typer
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from src.common.config import data_path, load_config, repo_root
from src.common.geo import pixel_area_km2, rasterize_geometries, read_vectors
from src.common.logging import get_logger
from src.common.metrics import Counts
from src.level1_radar.otsu import otsu_threshold
from src.level1_radar.terrain import iter_windows

log = get_logger("nepal_level1")
app = typer.Typer(add_completion=False, help=__doc__)
STEPS = ("terrain", "predict", "masks", "depth", "compare")
METHODS = ("model", "change_otsu", "otsu")
BLOCK = 4096


def _cfg(path: Path | None = None) -> dict[str, Any]:
    return load_config(path or repo_root() / "configs" / "nepal.yaml")


def scenes(cfg: dict[str, Any]) -> tuple[Path, Path | None]:
    """Post-flood scene for the first post date and its own pre-flood scene (if exported).

    The pre scene is ``S1_<post date>_pre.tif``: the same relative orbit as that post date
    (see ``scripts/nepal_s1_download.py export``), never another date's pre scene.
    """
    s1_dir = data_path(cfg["paths"]["s1_dir"])
    stem = f"S1_{cfg['dates']['post'][0].replace('-', '')}"
    post = s1_dir / f"{stem}.tif"
    if not post.exists():
        raise typer.Exit(f"{post} missing: run scripts/nepal_s1_download.py first")
    pre = s1_dir / f"{stem}_pre.tif"
    return post, (pre if pre.exists() else None)


def _profile(src: rasterio.io.DatasetReader, dtype: str, nodata: float | int) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": src.height,
        "width": src.width,
        "count": 1,
        "dtype": dtype,
        "crs": src.crs,
        "transform": src.transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


def read_aligned(
    path: Path,
    ref: rasterio.io.DatasetReader,
    window: Window,
    band: int = 1,
    resampling: str = "bilinear",
) -> np.ndarray:
    """Band of ``path`` on the reference grid's window (reprojected if grids differ)."""
    with rasterio.open(path) as src:
        if (src.crs, src.transform, src.shape) == (ref.crs, ref.transform, ref.shape):
            out = src.read(band, window=window).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                out[out == src.nodata] = np.nan
            return out
        out = np.full((int(window.height), int(window.width)), np.nan, np.float32)
        reproject(
            rasterio.band(src, band),
            out,
            dst_transform=window_transform(window, ref.transform),
            dst_crs=ref.crs,
            dst_nodata=np.nan,
            resampling=getattr(rasterio.enums.Resampling, resampling),
        )
        return out


def sample_values(path: Path, band: int, n_blocks: int = 40, per_block: int = 50000) -> np.ndarray:
    """Random finite values from random blocks (for scene-wide Otsu thresholds)."""
    rng = np.random.default_rng(0)
    with rasterio.open(path) as src:
        windows = list(iter_windows(src.height, src.width, 1024))
        picks = rng.choice(len(windows), size=min(n_blocks, len(windows)), replace=False)
        values = []
        for i in picks:
            block = src.read(band, window=windows[i]).astype(np.float32).ravel()
            block = block[np.isfinite(block)]
            if block.size:
                values.append(rng.choice(block, size=min(per_block, block.size), replace=False))
    return np.concatenate(values) if values else np.empty(0, np.float32)


# --------------------------------------------------------------------------- steps


def step_terrain(cfg: dict[str, Any], overwrite: bool) -> None:
    """HAND and slope on the post scene grid."""
    from src.level1_radar.terrain import write_terrain_rasters

    post, _ = scenes(cfg)
    out_dir = data_path(cfg["paths"]["terrain_dir"])
    if (out_dir / f"{post.stem}_HAND.tif").exists() and not overwrite:
        log.info("terrain exists, skipping")
        return
    write_terrain_rasters(post, out_dir)


def step_predict(cfg: dict[str, Any], overwrite: bool) -> None:
    """Ensemble flood probability and confidence."""
    from src.level1_radar.predict import Ensemble, member_paths, predict_scene

    post, _ = scenes(cfg)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    if (out_dir / f"{post.stem}_flood_prob.tif").exists() and not overwrite:
        log.info("prediction exists, skipping")
        return
    model_cfg = load_config(repo_root() / cfg["model"]["config"])
    ensemble = Ensemble.from_checkpoints(member_paths(model_cfg))
    terrain_dir = data_path(cfg["paths"]["terrain_dir"])
    terrain = {
        c: terrain_dir / f"{post.stem}_{c}.tif" for c in ensemble.channels if c in ("HAND", "SLOPE")
    }
    predict_scene(ensemble, post, out_dir, terrain)


def step_masks(cfg: dict[str, Any], overwrite: bool) -> None:
    """Write the three flood masks (uint8 1 flood / 0 dry / 255 no data)."""
    post, pre = scenes(cfg)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    outs = {m: out_dir / f"{post.stem}_{m}_mask.tif" for m in METHODS}
    if all(p.exists() for p in outs.values()) and not overwrite:
        log.info("masks exist, skipping")
        return
    prob_path = out_dir / f"{post.stem}_flood_prob.tif"
    hand_path = data_path(cfg["paths"]["terrain_dir"]) / f"{post.stem}_HAND.tif"
    jrc_path = data_path(cfg["paths"]["jrc"])
    vh_threshold = otsu_threshold(sample_values(post, 2))
    log.info("scene-wide VH Otsu threshold %.2f dB", vh_threshold)
    change_threshold = None
    if pre is not None:
        with rasterio.open(post) as a:
            rng_vals = []
            for w in list(iter_windows(a.height, a.width, 1024))[::25][:60]:
                diff = a.read(2, window=w).astype(np.float32) - read_aligned(pre, a, w, band=2)
                diff = diff[np.isfinite(diff)]
                if diff.size:
                    rng_vals.append(np.random.default_rng(0).choice(diff, min(50000, diff.size)))
        change_threshold = min(
            otsu_threshold(np.concatenate(rng_vals)), -abs(cfg["change_otsu"]["min_drop_db"])
        )
        log.info("change (after - before) VH threshold %.2f dB", change_threshold)
    occ_limit = cfg["permanent_water_occurrence"]
    max_hand = cfg["change_otsu"]["max_hand_m"]
    threshold = cfg["model"]["threshold"]
    with rasterio.open(post) as src:
        dsts = {m: rasterio.open(p, "w", **_profile(src, "uint8", 255)) for m, p in outs.items()}
        try:
            for window in iter_windows(src.height, src.width, BLOCK):
                vv_vh = src.read([1, 2], window=window).astype(np.float32)
                no_data = np.isnan(vv_vh).all(axis=0)
                jrc = read_aligned(jrc_path, src, window) if jrc_path.exists() else None
                permanent = np.zeros(no_data.shape, bool) if jrc is None else (jrc >= occ_limit)
                hand = read_aligned(hand_path, src, window) if hand_path.exists() else None
                masks: dict[str, np.ndarray] = {}
                if prob_path.exists():
                    masks["model"] = read_aligned(prob_path, src, window) >= threshold
                with np.errstate(invalid="ignore"):
                    masks["otsu"] = vv_vh[1] < vh_threshold
                    if change_threshold is not None:
                        diff = vv_vh[1] - read_aligned(pre, src, window, band=2)
                        change = diff < change_threshold
                        if hand is not None:
                            change &= ~(hand > max_hand)
                        masks["change_otsu"] = change
                for method, dst in dsts.items():
                    if method not in masks:
                        continue
                    out = (masks[method] & ~permanent).astype(np.uint8)
                    out[no_data] = 255
                    dst.write(out[None], window=window)
        finally:
            for dst in dsts.values():
                dst.close()
    for p in outs.values():
        log.info("wrote %s", p)


def step_depth(cfg: dict[str, Any], overwrite: bool) -> None:
    """FwDET depth of the model flood mask."""
    from src.common.raster import write_raster
    from src.depth.depth_map import depth_from_flood
    from src.depth.fwdet import FwdetSettings

    post, _ = scenes(cfg)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    out = out_dir / f"{post.stem}_depth.tif"
    if out.exists() and not overwrite:
        log.info("depth exists, skipping")
        return
    depth_cfg = load_config(repo_root() / "configs" / "depth.yaml")
    mask = out_dir / f"{post.stem}_model_mask.tif"
    depth, grid = depth_from_flood(
        mask,
        FwdetSettings(**depth_cfg["fwdet"]),
        is_mask=True,
        resolution_m=cfg["depth"]["resolution_m"],
    )
    write_raster(
        out,
        depth,
        crs=grid.crs,
        transform=grid.transform,
        nodata=float("nan"),
        band_names=["water_depth_m"],
    )
    log.info("wrote %s", out)


def _components(mask: np.ndarray, area_row: np.ndarray, top: int) -> list[tuple[float, int, int]]:
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(
        np.repeat(area_row[:, None], mask.shape[1], 1), labels, index=np.arange(1, n + 1)
    )
    centres = ndimage.center_of_mass(mask, labels, np.arange(1, n + 1))
    order = np.argsort(areas)[::-1][:top]
    return [(float(areas[i]), int(centres[i][0]), int(centres[i][1])) for i in order]


def step_compare(cfg: dict[str, Any], overwrite: bool) -> None:
    """Areas and agreement with UNOSAT inside the analysed area; figures; report."""
    import geopandas as gpd

    post, _ = scenes(cfg)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    vectors = read_vectors(data_path(cfg["paths"]["unosat_dir"]))
    is_extent = vectors["source"].str.contains("extent|analys|aoi", case=False)
    flood_polys = vectors[~is_extent]
    extent_polys = vectors[is_extent]
    results: dict[str, Any] = {"scene": post.name, "methods": {}, "failure_cases": {}}
    counts = {m: Counts() for m in METHODS}
    area = {m: 0.0 for m in METHODS}
    unosat_area = 0.0
    analysed_area = 0.0
    patches: dict[str, list] = {m: [] for m in METHODS}
    with rasterio.open(post) as src:
        flood_g = gpd.GeoSeries(flood_polys.geometry, crs="EPSG:4326").to_crs(src.crs)
        extent_g = (
            gpd.GeoSeries(extent_polys.geometry, crs="EPSG:4326").to_crs(src.crs)
            if len(extent_polys)
            else None
        )
        for window in iter_windows(src.height, src.width, BLOCK):
            wt = window_transform(window, src.transform)
            shape = (int(window.height), int(window.width))
            valid = np.isfinite(src.read(1, window=window).astype(np.float32))
            if extent_g is not None:
                valid &= rasterize_geometries(extent_g, shape, wt)
            if not valid.any():
                continue
            area_row = pixel_area_km2(src.crs, wt, shape[0])
            ref = rasterize_geometries(flood_g, shape, wt) & valid
            unosat_area += float((ref.sum(axis=1) * area_row).sum())
            analysed_area += float((valid.sum(axis=1) * area_row).sum())
            for m in METHODS:
                path = out_dir / f"{post.stem}_{m}_mask.tif"
                if not path.exists():
                    continue
                with rasterio.open(path) as mask_src:
                    pred = (mask_src.read(1, window=window) == 1) & valid
                tp = pred & ref
                counts[m] = counts[m] + Counts(
                    int(tp.sum()), int((pred & ~ref).sum()), int((~pred & ref).sum()), 0
                )
                area[m] += float((pred.sum(axis=1) * area_row).sum())
                for kind, wrong in (("false_flood", pred & ~ref), ("missed_flood", ~pred & ref)):
                    for a, r, c in _components(wrong, area_row, top=3):
                        x, y = rasterio.transform.xy(wt, r, c)
                        patches[m].append(
                            {
                                "type": kind,
                                "area_km2": round(a, 3),
                                "lon": round(x, 4),
                                "lat": round(y, 4),
                            }
                        )
    results["unosat_flood_km2"] = round(unosat_area, 2)
    results["analysed_area_km2"] = round(analysed_area, 1)
    for m in METHODS:
        c = counts[m]
        results["methods"][m] = {
            "flooded_km2": round(area[m], 2),
            **{
                k: round(v, 4)
                for k, v in c.as_dict().items()
                if k in ("iou", "f1", "precision", "recall")
            },
        }
        results["failure_cases"][m] = sorted(patches[m], key=lambda p: -p["area_km2"])[:8]
    path = repo_root() / "docs" / "results" / "nepal_level1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    make_figures(cfg, post)
    write_doc(results)


def make_figures(cfg: dict[str, Any], post: Path, max_px: int = 1600) -> None:
    """Overview PNGs (decimated): VH, flood probability, confidence, masks, depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = data_path(cfg["paths"]["level1_dir"])
    panels = [
        (post, 2, "post-flood VH (dB)", "gray", (-30, -5)),
        (out_dir / f"{post.stem}_flood_prob.tif", 1, "flood probability", "Blues", (0, 1)),
        (
            out_dir / f"{post.stem}_flood_std.tif",
            1,
            "ensemble std (low confidence)",
            "magma",
            (0, 0.3),
        ),
        (out_dir / f"{post.stem}_model_mask.tif", 1, "model flood mask", "Blues", (0, 1)),
        (
            out_dir / f"{post.stem}_change_otsu_mask.tif",
            1,
            "change-detection Otsu",
            "Blues",
            (0, 1),
        ),
        (out_dir / f"{post.stem}_depth.tif", 1, "water depth (m)", "viridis", (0, 3)),
    ]
    panels = [p for p in panels if p[0].exists()]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (path, band, title, cmap, (lo, hi)) in zip(axes.ravel(), panels, strict=False):
        with rasterio.open(path) as src:
            factor = max(1, max(src.width, src.height) // max_px)
            img = src.read(band, out_shape=(src.height // factor, src.width // factor)).astype(
                float
            )
            if src.nodata is not None and not np.isnan(src.nodata):
                img[img == src.nodata] = np.nan
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    out = repo_root() / "docs" / "figures" / "nepal_level1_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    log.info("wrote %s", out)


def write_doc(results: dict[str, Any]) -> None:
    """docs/nepal_level1.md with the comparison table and the failure cases to inspect."""
    lines = [
        "# Nepal replay, Level 1 (Koshi floods, 27 September 2024)",
        "",
        "Generated by `python scripts/nepal_level1.py run`. Reference: UNOSAT flood extent of",
        "27 September 2024 (reported flooded area about 50 km^2). Permanent water (JRC",
        "occurrence >= 50 %) is removed from every method before comparing.",
        "",
        f"Scene: `{results['scene']}`; analysed area {results['analysed_area_km2']} km^2; "
        f"UNOSAT flood inside it {results['unosat_flood_km2']} km^2.",
        "",
        "| Method | Flooded km^2 | IoU | F1 | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "model": "E6 ensemble (U-Net, VV+VH+HAND)",
        "change_otsu": "Change-detection Otsu",
        "otsu": "Single-date Otsu (VH)",
    }
    for m, r in results["methods"].items():
        lines.append(
            f"| {names[m]} | {r['flooded_km2']} | {r['iou']} | {r['f1']} | "
            f"{r['precision']} | {r['recall']} |"
        )
    lines += [
        "",
        "![overview](figures/nepal_level1_overview.png)",
        "",
        "## Largest disagreements (failure cases to inspect)",
        "",
        "Open these places in QGIS over the S1 scene and the UNOSAT polygons and write",
        "down the cause (radar shadow, wet soil, vegetation, missed narrow channel, UNOSAT",
        "edge, ...).",
        "",
        "| Method | Type | km^2 | lon | lat | Cause |",
        "|---|---|---|---|---|---|",
    ]
    for m, cases in results["failure_cases"].items():
        for case in cases:
            lines.append(
                f"| {m} | {case['type']} | {case['area_km2']} | {case['lon']} | {case['lat']} | |"
            )
    path = repo_root() / "docs" / "nepal_level1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", path)


@app.callback()
def _group() -> None:
    """Nepal Level 1 replay (see the module docstring)."""


@app.command()
def run(
    step: list[str] = typer.Option(list(STEPS), help=f"Steps to run: {', '.join(STEPS)}"),
    overwrite: bool = typer.Option(False, help="Recompute outputs that already exist."),
    config: Path = typer.Option(None, help="Default: configs/nepal.yaml"),
) -> None:
    """Run the replay steps in order."""
    cfg = _cfg(config)
    functions = {
        "terrain": step_terrain,
        "predict": step_predict,
        "masks": step_masks,
        "depth": step_depth,
        "compare": step_compare,
    }
    for name in STEPS:
        if name in step:
            log.info("== %s", name)
            functions[name](cfg, overwrite)


if __name__ == "__main__":
    app()
