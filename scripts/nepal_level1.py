"""Nepal radar replay for the Koshi floods, 27 September and 1 October 2024 (prompt A10).

Run after ``scripts/nepal_s1_download.py`` (scenes in ``$DATA_ROOT/nepal/s1``) and after the
UNOSAT product FL20240928NPL was unzipped by hand into ``$DATA_ROOT/nepal/unosat/``:

    python scripts/nepal_level1.py run            # every step for every post date
    python scripts/nepal_level1.py run --date 2024-10-01
    python scripts/nepal_level1.py run --step compare --overwrite

Every step runs once per post date (``dates.post``); outputs are named after the scene, so the
dates never mix. Each date is compared with the UNOSAT layers of the same date only
(``src/common/unosat.py``). Credit: "Analysis: UNOSAT".

Steps (``--step`` runs one):

1. ``terrain``  HAND and slope on the post-flood scene grid (scripts/prepare_hand.py logic)
2. ``predict``  Level 1 ensemble (configs/E6ens.yaml) -> flood probability + confidence (std)
3. ``masks``    flood masks of three methods, without permanent water (JRC occurrence >= 50 %):
                ``model`` (probability >= 0.5), ``change_otsu`` (after - before VH, Nepal only,
                HAND <= 15 m) and ``otsu`` (one VH threshold for the whole post scene)
4. ``depth``    FwDET depth of the model flood mask on a 30 m UTM grid
5. ``compare``  flooded km^2 and IoU / F1 / precision / recall against the UNOSAT flood extent
                of the same date, inside that date's UNOSAT analysis extent, with permanent
                water (JRC) left out; also for the Nepal part alone (the extents cover India
                too). Largest disagreement patches (failure cases to inspect), PNG maps,
                ``docs/results/nepal_level1_<YYYYMMDD>.json`` and ``docs/nepal_level1.md``

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
from src.common.geo import pixel_area_km2, rasterize_geometries
from src.common.logging import get_logger
from src.common.metrics import Counts
from src.level1_radar.otsu import otsu_threshold
from src.level1_radar.terrain import iter_windows

log = get_logger("nepal_level1")
app = typer.Typer(add_completion=False, help=__doc__)
STEPS = ("terrain", "predict", "masks", "depth", "compare")
METHODS = ("model", "model_new", "change_otsu", "otsu")
BLOCK = 4096


def _cfg(path: Path | None = None) -> dict[str, Any]:
    return load_config(path or repo_root() / "configs" / "nepal.yaml")


def scenes(cfg: dict[str, Any], day: str | None = None) -> tuple[Path, Path | None]:
    """Post-flood scene of ``day`` (default: the first post date) and its own pre scene.

    The pre scene is ``S1_<post date>_pre.tif``: the same relative orbit as that post date
    (see ``scripts/nepal_s1_download.py export``), never another date's pre scene.
    """
    s1_dir = data_path(cfg["paths"]["s1_dir"])
    stem = f"S1_{(day or cfg['dates']['post'][0]).replace('-', '')}"
    post = s1_dir / f"{stem}.tif"
    if not post.exists():
        raise typer.Exit(f"{post} missing: run scripts/nepal_s1_download.py first")
    pre = s1_dir / f"{stem}_pre.tif"
    return post, (pre if pre.exists() else None)


def _same_grid(a: Path, b: Path) -> bool:
    with rasterio.open(a) as x, rasterio.open(b) as y:
        return (x.crs, x.transform, x.shape) == (y.crs, y.transform, y.shape)


def terrain_file(cfg: dict[str, Any], post: Path, kind: str) -> Path:
    """``<post stem>_<KIND>.tif``, or another date's terrain file on exactly the same grid.

    Terrain does not change between dates; the Nepal exports of every post date share one grid,
    so the 1 October scene reuses the 27 September HAND and slope instead of a 2 GB copy.
    """
    terrain_dir = data_path(cfg["paths"]["terrain_dir"])
    own = terrain_dir / f"{post.stem}_{kind.upper()}.tif"
    if own.exists():
        return own
    for other in cfg["dates"]["post"]:
        other_post = post.with_name(f"S1_{other.replace('-', '')}.tif")
        candidate = terrain_dir / f"{other_post.stem}_{kind.upper()}.tif"
        if (
            other_post != post
            and candidate.exists()
            and other_post.exists()
            and _same_grid(other_post, post)
        ):
            return candidate
    return own


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


def step_terrain(cfg: dict[str, Any], overwrite: bool, day: str) -> None:
    """HAND and slope on the post scene grid (each skipped if it, or a same-grid copy, exists).

    Run ``python scripts/prepare_hand.py tiles <post scene>`` first to read the terrain tiles
    from disk instead of streaming them.
    """
    from src.level1_radar.terrain import write_terrain_rasters

    post, _ = scenes(cfg, day)
    out_dir = data_path(cfg["paths"]["terrain_dir"])
    kinds = tuple(
        k for k in ("hand", "slope") if overwrite or not terrain_file(cfg, post, k).exists()
    )
    if not kinds:
        log.info("terrain exists, skipping")
        return
    write_terrain_rasters(post, out_dir, kinds=kinds)


def step_predict(cfg: dict[str, Any], overwrite: bool, day: str) -> None:
    """Ensemble flood probability and confidence for the post scene and its pre scene.

    The pre scene (same relative orbit, same grid) gets the same model and the post scene's
    terrain, for the ``model_new`` (new water) mask.
    """
    from src.level1_radar.predict import Ensemble, member_paths, predict_scene

    post, pre = scenes(cfg, day)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    todo = [
        s
        for s in (post, pre)
        if s is not None and (overwrite or not (out_dir / f"{s.stem}_flood_prob.tif").exists())
    ]
    if pre is None:
        log.warning("no pre scene for %s: the new-water mask is skipped", day)
    if not todo:
        log.info("predictions exist, skipping")
        return
    model_cfg = load_config(repo_root() / cfg["model"]["config"])
    ensemble = Ensemble.from_checkpoints(member_paths(model_cfg))
    terrain = {c: terrain_file(cfg, post, c) for c in ensemble.channels if c in ("HAND", "SLOPE")}
    for scene in todo:
        if scene is pre and not _same_grid(pre, post):
            raise typer.Exit(f"{pre.name} is not on the grid of {post.name}")
        log.info("predicting %s", scene.name)
        predict_scene(ensemble, scene, out_dir, terrain)


def step_masks(cfg: dict[str, Any], overwrite: bool, day: str) -> None:
    """Write the flood masks (uint8 1 flood / 0 dry / 255 no data), one per available method.

    ``model``: post probability >= threshold (all water). ``model_new``: water in post and not in
    pre (``src/level1_radar/new_water.py``); no data where the pre scene has none. JRC permanent
    water is removed from every mask.
    """
    from src.level1_radar.new_water import new_water

    post, pre = scenes(cfg, day)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    prob_path = out_dir / f"{post.stem}_flood_prob.tif"
    pre_prob_path = out_dir / f"{pre.stem}_flood_prob.tif" if pre is not None else None
    available = {
        "model": prob_path.exists(),
        "model_new": prob_path.exists() and pre_prob_path is not None and pre_prob_path.exists(),
        "change_otsu": pre is not None,
        "otsu": True,
    }
    outs = {m: out_dir / f"{post.stem}_{m}_mask.tif" for m in METHODS if available[m]}
    for m in METHODS:
        if not available[m]:
            log.warning("%s mask skipped for %s (missing inputs)", m, day)
    if all(p.exists() for p in outs.values()) and not overwrite:
        log.info("masks exist, skipping")
        return
    hand_path = terrain_file(cfg, post, "hand")
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
                extra_no_data: dict[str, np.ndarray] = {}
                if "model" in outs:
                    prob = read_aligned(prob_path, src, window)
                    with np.errstate(invalid="ignore"):
                        masks["model"] = prob >= threshold
                    if "model_new" in outs:
                        masks["model_new"], extra_no_data["model_new"] = new_water(
                            prob, read_aligned(pre_prob_path, src, window), threshold
                        )
                with np.errstate(invalid="ignore"):
                    masks["otsu"] = vv_vh[1] < vh_threshold
                    if change_threshold is not None:
                        diff = vv_vh[1] - read_aligned(pre, src, window, band=2)
                        change = diff < change_threshold
                        if hand is not None:
                            change &= ~(hand > max_hand)
                        masks["change_otsu"] = change
                for method, dst in dsts.items():
                    out = (masks[method] & ~permanent).astype(np.uint8)
                    out[no_data] = 255
                    if method in extra_no_data:
                        out[extra_no_data[method] & ~no_data] = 255
                    dst.write(out[None], window=window)
        finally:
            for dst in dsts.values():
                dst.close()
    for p in outs.values():
        log.info("wrote %s", p)


def step_depth(cfg: dict[str, Any], overwrite: bool, day: str) -> None:
    """FwDET depth of the model flood mask."""
    from src.common.raster import write_raster
    from src.depth.depth_map import depth_from_flood
    from src.depth.fwdet import FwdetSettings

    post, _ = scenes(cfg, day)
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


REGIONS = ("unosat_extent", "nepal")
# JRC occurrence (%) bins for the false-flood breakdown (below the permanent-water limit)
JRC_BINS = {"never water (0)": (0, 1), "rare (1-9)": (1, 10), "seasonal (10-49)": (10, 50)}
REGION_NOTES = {
    "unosat_extent": "UNOSAT analysis extent of the same date, inside our scene",
    "nepal": "the same, clipped to Nepal (the extent polygons also cover India)",
}


def _result_path(post: Path) -> Path:
    return repo_root() / "docs" / "results" / f"nepal_level1_{post.stem.removeprefix('S1_')}.json"


def _nepal_outline(cfg: dict[str, Any], crs: Any) -> Any:
    """Union of the Nepal local-level polygons in ``crs`` (None if the file is missing)."""
    import geopandas as gpd

    path = data_path(cfg["paths"]["admin"])
    if not path.exists():
        log.warning("%s missing: the Nepal-only comparison is skipped", path)
        return None
    return gpd.read_file(path).to_crs(crs).union_all()


def score_region(
    pred: np.ndarray, ref: np.ndarray, region: np.ndarray
) -> tuple[Counts, np.ndarray, np.ndarray]:
    """Counts of ``pred`` against ``ref`` inside ``region``, and both masks clipped to it."""
    p, r = pred & region, ref & region
    return Counts(int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum()), 0), p, r


def step_compare(cfg: dict[str, Any], overwrite: bool, day: str) -> None:
    """Agreement with the UNOSAT flood extent of the same date, inside its analysis extent.

    Permanent water (JRC occurrence >= ``permanent_water_occurrence``) is removed from the
    analysed area, so it counts neither for our masks nor for UNOSAT. Results are given for the
    whole analysis extent (inside our scene) and for its Nepal part, which is what UNOSAT's own
    table covers (``unosat.reference`` in the config, used as a check).
    """
    import geopandas as gpd

    from src.common.unosat import CREDIT, find_layers

    post, _ = scenes(cfg, day)
    out_dir = data_path(cfg["paths"]["level1_dir"])
    layers = find_layers(data_path(cfg["paths"]["unosat_dir"]), day)
    if layers.flood is None or layers.extent is None:
        log.warning("no UNOSAT flood and analysis-extent layers dated %s; skipping", day)
        return
    log.info("UNOSAT %s: %s, %s", day, layers.flood.name, layers.extent.name)
    jrc_path = data_path(cfg["paths"]["jrc"])
    if not jrc_path.exists():
        raise typer.Exit(f"{jrc_path} missing: permanent water cannot be left out")
    occ_limit = cfg["permanent_water_occurrence"]
    names = list(REGIONS)
    counts = {(r, m): Counts() for r in names for m in METHODS}
    area = {(r, m): 0.0 for r in names for m in METHODS}
    unosat_area = dict.fromkeys(names, 0.0)
    unosat_area_with_jrc = dict.fromkeys(names, 0.0)
    analysed_area = dict.fromkeys(names, 0.0)
    patches: dict[str, list] = {m: [] for m in METHODS}
    false_by_jrc = {m: dict.fromkeys(JRC_BINS, 0.0) for m in METHODS}
    mask_paths = {m: out_dir / f"{post.stem}_{m}_mask.tif" for m in METHODS}
    mask_paths = {m: p for m, p in mask_paths.items() if p.exists()}
    if not mask_paths:
        raise typer.Exit(f"no flood masks for {day}: run the masks step first")
    with rasterio.open(post) as src:
        flood_g = gpd.read_file(layers.flood).to_crs(src.crs).geometry
        extent_g = gpd.read_file(layers.extent).to_crs(src.crs).geometry
        nepal = _nepal_outline(cfg, src.crs)
        if nepal is None:
            names = ["unosat_extent"]
        for window in iter_windows(src.height, src.width, BLOCK):
            wt = window_transform(window, src.transform)
            shape = (int(window.height), int(window.width))
            inside = np.isfinite(src.read(1, window=window).astype(np.float32))
            inside &= rasterize_geometries(extent_g, shape, wt)
            if not inside.any():
                continue
            jrc = read_aligned(jrc_path, src, window, resampling="nearest")
            permanent = jrc >= occ_limit
            ref = rasterize_geometries(flood_g, shape, wt)
            # each region with and without permanent water (the latter only to report areas)
            with_jrc = {"unosat_extent": inside}
            if nepal is not None:
                with_jrc["nepal"] = inside & rasterize_geometries([nepal], shape, wt)
            region_masks = {r: m & ~permanent for r, m in with_jrc.items()}
            area_row = pixel_area_km2(src.crs, wt, shape[0])

            def km2(mask: np.ndarray, rows: np.ndarray = area_row) -> float:
                return float((mask.sum(axis=1) * rows).sum())

            for r in names:
                region = region_masks[r]
                analysed_area[r] += km2(region)
                unosat_area[r] += km2(ref & region)
                unosat_area_with_jrc[r] += km2(ref & with_jrc[r])
            for m, path in mask_paths.items():
                with rasterio.open(path) as mask_src:
                    pred = mask_src.read(1, window=window) == 1
                for r in names:
                    c, p, rr = score_region(pred, ref, region_masks[r])
                    counts[(r, m)] = counts[(r, m)] + c
                    area[(r, m)] += km2(p)
                    if r != "unosat_extent":
                        continue
                    for label, (lo, hi) in JRC_BINS.items():
                        in_bin = p & ~rr & (jrc >= lo) & (jrc < hi)
                        false_by_jrc[m][label] += km2(in_bin)
                    for kind, wrong in (("false_flood", p & ~rr), ("missed_flood", ~p & rr)):
                        for a, row, col in _components(wrong, area_row, top=3):
                            x, y = rasterio.transform.xy(wt, row, col)
                            patches[m].append(
                                {
                                    "type": kind,
                                    "area_km2": round(a, 3),
                                    "lon": round(x, 4),
                                    "lat": round(y, 4),
                                }
                            )
    reference = (cfg.get("unosat") or {}).get("reference", {}).get(day)
    results: dict[str, Any] = {
        "date": day,
        "scene": post.name,
        "credit": CREDIT,
        "unosat_flood_layer": layers.flood.name,
        "unosat_extent_layer": layers.extent.name,
        "permanent_water": f"JRC occurrence >= {occ_limit} %, removed from the analysed area",
        "unosat_table": reference,
        "regions": {},
        "false_flood_km2_by_jrc_occurrence": {
            m: {k: round(v, 1) for k, v in false_by_jrc[m].items()} for m in mask_paths
        },
        "failure_cases": {
            m: sorted(patches[m], key=lambda p: -p["area_km2"])[:8] for m in mask_paths
        },
    }
    for r in names:
        results["regions"][r] = {
            "note": REGION_NOTES[r],
            "analysed_km2": round(analysed_area[r], 1),
            "unosat_flood_km2": round(unosat_area[r], 2),
            "unosat_flood_km2_before_jrc_removal": round(unosat_area_with_jrc[r], 2),
            "methods": {
                m: {
                    "flooded_km2": round(area[(r, m)], 2),
                    **{
                        k: round(v, 4)
                        for k, v in counts[(r, m)].as_dict().items()
                        if k in ("iou", "f1", "precision", "recall")
                    },
                }
                for m in mask_paths
            },
        }
    path = _result_path(post)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    make_figures(cfg, post)
    write_doc(cfg)


def make_figures(cfg: dict[str, Any], post: Path, max_px: int = 1600) -> None:
    """Overview PNG of one date (decimated): VH, flood probability, confidence, masks, depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.common.unosat import CREDIT

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
        (
            out_dir / f"{post.stem}_pre_flood_prob.tif",
            1,
            "pre-scene water probability",
            "Blues",
            (0, 1),
        ),
        (out_dir / f"{post.stem}_model_mask.tif", 1, "(a) all water - JRC", "Blues", (0, 1)),
        (out_dir / f"{post.stem}_model_new_mask.tif", 1, "(b) new water", "Blues", (0, 1)),
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
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
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
    day = post.stem.removeprefix("S1_")
    fig.suptitle(f"Nepal replay {day}. Reference flood extent for comparison: {CREDIT}")
    fig.tight_layout()
    out = repo_root() / "docs" / "figures" / f"nepal_level1_overview_{day}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    log.info("wrote %s", out)


METHOD_NAMES = {
    "model": "(a) E6 ensemble, all water minus JRC >= 50 %",
    "model_new": "(b) E6 ensemble, new water (post and not pre)",
    "change_otsu": "Change-detection Otsu",
    "otsu": "Single-date Otsu (VH)",
}


def write_doc(cfg: dict[str, Any]) -> None:
    """docs/nepal_level1.md: one section per post date that has comparison results."""
    from src.common.unosat import CREDIT

    unosat = cfg.get("unosat") or {}
    lines = [
        "# Nepal replay, Level 1 (Koshi floods, September to October 2024)",
        "",
        "Generated by `python scripts/nepal_level1.py run`. Each of our Sentinel-1 scenes is",
        "compared with the UNOSAT flood extent of the same date "
        f"(product {unosat.get('product')}),",
        "inside UNOSAT's analysis extent of that date. Permanent water (JRC occurrence >= "
        f"{cfg['permanent_water_occurrence']} %) is left out of the analysed area for every map,",
        "UNOSAT's included: the product has no permanent-water layer of its own.",
        "",
        f"**{CREDIT}** (United Nations Satellite Centre, FL20240928NPL, preliminary, not",
        "field-validated). Contains modified Copernicus Sentinel data [2024].",
        "",
        "## Two versions of our flood map",
        "",
        "- **(a) All water minus JRC.** The E6 ensemble finds every pixel of open water on the",
        "  post date; water JRC saw in at least "
        f"{cfg['permanent_water_occurrence']} % of its observations is removed.",
        "- **(b) New water.** The same ensemble also runs on the pre-flood scene of the same",
        "  relative orbit (`S1_<post date>_pre.tif`, the latest acquisition in "
        f"{' to '.join(cfg['dates'].get('pre_window', ['the pre-flood window']))}),",
        "  and flood is water in the post scene that is not water in the pre scene. JRC",
        "  permanent water is removed as in (a). (b) is the map used for exposure and priority",
        "  (`configs/priority.yaml`), so the normal river channel does not count as exposed.",
        "",
        'UNOSAT\'s layers are "New Water / Water Increase" against **its own reference image,',
        "Sentinel-1 of 11 June 2024**, which is not part of the product. Our pre scenes are from",
        "**mid-September 2024**, deep in the monsoon, when much more of the floodplain is already",
        "wet. Water that appeared between June and mid-September counts as new for UNOSAT but",
        "not for (b), so **(b) may under-count relative to UNOSAT**; (a) over-counts because it",
        "keeps all seasonal water. The two bracket UNOSAT's definition.",
        "",
    ]
    findings, recalls = [], []
    for day in cfg["dates"]["post"]:
        path = _result_path(Path(f"S1_{day.replace('-', '')}.tif"))
        if path.exists():
            res = json.loads(path.read_text(encoding="utf-8"))
            by_jrc = res.get("false_flood_km2_by_jrc_occurrence", {}).get("model")
            whole = res["regions"]["unosat_extent"]
            methods = whole["methods"]
            if "model" in methods and "model_new" in methods:
                a, b = methods["model"], methods["model_new"]
                recalls.append(
                    f"| {day} | {whole['unosat_flood_km2']} | {a['flooded_km2']} | {a['recall']} | "
                    f"{b['flooded_km2']} | {b['recall']} | {b['precision']} |"
                )
            if by_jrc:
                findings.append(
                    f"| {day} | {sum(by_jrc.values()):.1f} | "
                    + " | ".join(f"{v:.1f}" for v in by_jrc.values())
                    + " |"
                )
    if findings:
        lines += [
            "## Seasonal water: why (a) disagrees with UNOSAT",
            "",
            "Area that (a) marks as flood but UNOSAT does not, inside the analysis extent, split",
            "by JRC water occurrence (km^2):",
            "",
            "| Date | False flood | " + " | ".join(f"JRC {k}" for k in JRC_BINS) + " |",
            "|---" * (2 + len(JRC_BINS)) + "|",
            *findings,
            "",
            "Most of it is water JRC has seen before, the larger part seasonal (occurrence 10 to",
            "49 %): the braided Koshi channels, sand bars and floodplain ponds that are wet every",
            "monsoon. JRC >= 50 % does not remove them, but UNOSAT's June 2024 reference does.",
            "(a)'s recall is high, so the extra area is mostly a difference of definition, not",
            "missed detections; (b) removes water already present in mid-September.",
            "",
        ]
    if recalls:
        lines += [
            "## New water (b): low recall against UNOSAT",
            "",
            "| Date | UNOSAT km^2 | (a) km^2 | (a) recall | (b) km^2 | (b) recall "
            "| (b) precision |",
            "|---|---|---|---|---|---|---|",
            *recalls,
            "",
            "(b) removes nearly all of (a)'s seasonal false flood (see its false-flood split in",
            "`docs/results/nepal_level1_<date>.json`), but it also loses most of UNOSAT's flood:",
            "the gap between (a)'s and (b)'s recall is UNOSAT flood that our model already saw as",
            "water in the mid-September pre scene. This is the expected direction (UNOSAT's",
            "reference is June 2024, ours is mid-September), but the size is large: by this",
            "comparison, most of what UNOSAT reports as new water was already wet before",
            "27 September. (b) therefore under-counts flood relative to UNOSAT, and exposure",
            "computed from (b) under-counts accordingly. Precision does not rise with (b): most of",
            "its new water is not in UNOSAT's layer either (open these places in QGIS; candidates",
            "are wet soil, new sand bars and changes between orbits).",
            "",
        ]
    for day in cfg["dates"]["post"]:
        path = _result_path(Path(f"S1_{day.replace('-', '')}.tif"))
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        table = res.get("unosat_table") or {}
        lines += [
            f"## {day}",
            "",
            f"Scene `{res['scene']}`; UNOSAT layers `{res['unosat_flood_layer']}` and "
            f"`{res['unosat_extent_layer']}`"
            + (f" (acquired {table['acquired_utc']} UTC)." if table.get("acquired_utc") else "."),
            "",
        ]
        for region, r in res["regions"].items():
            lines += [
                f"### Inside {r['note']}",
                "",
                f"Analysed area {r['analysed_km2']} km^2; UNOSAT flood inside it "
                f"{r['unosat_flood_km2']} km^2 ({r['unosat_flood_km2_before_jrc_removal']} km^2 "
                "before removing permanent water).",
            ]
            if region == "nepal" and table:
                lines.append(
                    f"UNOSAT's own table for this date: analysed {table['analysed_km2']} km^2, "
                    f"flood {table['flood_km2']} km^2, about {table['exposed']:,} people exposed."
                )
            lines += [
                "",
                "| Method | Flooded km^2 | IoU | F1 | Precision | Recall |",
                "|---|---|---|---|---|---|",
            ]
            for m, s in r["methods"].items():
                lines.append(
                    f"| {METHOD_NAMES[m]} | {s['flooded_km2']} | {s['iou']} | {s['f1']} | "
                    f"{s['precision']} | {s['recall']} |"
                )
            lines.append("")
        lines += [
            f"![overview {day}](figures/nepal_level1_overview_{day.replace('-', '')}.png)",
            "",
            "Largest disagreements inside the analysis extent (failure cases to inspect: open",
            "them in QGIS over the S1 scene and the UNOSAT polygons and write down the cause, such",
            "as radar shadow, wet soil, vegetation, a missed narrow channel or a UNOSAT edge).",
            "",
            "| Method | Type | km^2 | lon | lat | Cause |",
            "|---|---|---|---|---|---|",
        ]
        for m, cases in res["failure_cases"].items():
            for case in cases:
                lines.append(
                    f"| {m} | {case['type']} | {case['area_km2']} | {case['lon']} | "
                    f"{case['lat']} | |"
                )
        lines.append("")
    path = repo_root() / "docs" / "nepal_level1.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)


@app.callback()
def _group() -> None:
    """Nepal Level 1 replay (see the module docstring)."""


@app.command()
def run(
    step: list[str] = typer.Option(list(STEPS), help=f"Steps to run: {', '.join(STEPS)}"),
    overwrite: bool = typer.Option(False, help="Recompute outputs that already exist."),
    config: Path = typer.Option(None, help="Default: configs/nepal.yaml"),
    date: list[str] = typer.Option([], help="Post dates to run (default: all dates.post)."),
) -> None:
    """Run the replay steps in order, for each post date in turn."""
    cfg = _cfg(config)
    days = date or list(cfg["dates"]["post"])
    unknown = [d for d in days if d not in cfg["dates"]["post"]]
    if unknown:
        raise typer.BadParameter(f"not in dates.post: {', '.join(unknown)}")
    functions = {
        "terrain": step_terrain,
        "predict": step_predict,
        "masks": step_masks,
        "depth": step_depth,
        "compare": step_compare,
    }
    for day in days:
        for name in STEPS:
            if name in step:
                log.info("== %s %s", day, name)
                functions[name](cfg, overwrite, day)


if __name__ == "__main__":
    app()
