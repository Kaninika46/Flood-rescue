"""Build HAND and slope rasters on exactly the grid of each chip or of any GeoTIFF (prompt A2).

    # every chip listed in the split files (hand-labelled and weak), written to
    # $DATA_ROOT/sen1floods11/HAND/<chip>_HAND.tif and .../SLOPE/<chip>_SLOPE.tif
    python scripts/prepare_hand.py chips
    python scripts/prepare_hand.py chips --split s11_bolivia --workers 4

    # any GeoTIFF, e.g. a Nepal Sentinel-1 scene (processed in blocks, any size)
    python scripts/prepare_hand.py raster $DATA_ROOT/nepal/s1/S1_20240927.tif \
        --out-dir $DATA_ROOT/nepal/terrain

    # PNG: S1 VH, label and HAND side by side for three chips
    python scripts/prepare_hand.py figure Bolivia_103757 India_1018327 Sri-Lanka_120804

Tiles are streamed from the public buckets; add ``--cache-dir $DATA_ROOT/terrain_tiles`` to keep
whole tiles locally when the same area is processed many times.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import typer

from src.common.config import data_path, repo_root
from src.common.logging import get_logger
from src.common.raster import read_raster, write_raster
from src.level1_radar.s11 import read_split
from src.level1_radar.terrain import (
    Grid,
    hand_for_grid,
    slope_for_grid,
    write_terrain_rasters,
)

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger("prepare_hand")
DEFAULT_SPLITS = ["s11_train", "s11_val", "s11_test", "s11_bolivia", "s11_weak"]


def build_one(s1_path: Path, hand_path: Path, slope_path: Path, cache_dir: Path | None) -> str:
    """HAND and slope for one chip; skips files that already exist."""
    grid = Grid.of(s1_path)
    if not hand_path.exists():
        hand = hand_for_grid(grid, cache_dir)
        write_raster(
            hand_path,
            hand,
            crs=grid.crs,
            transform=grid.transform,
            nodata=float("nan"),
            band_names=["HAND_m"],
        )
    if not slope_path.exists():
        slope = slope_for_grid(grid, cache_dir)
        write_raster(
            slope_path,
            slope,
            crs=grid.crs,
            transform=grid.transform,
            nodata=float("nan"),
            band_names=["slope_deg"],
        )
    return s1_path.name


@app.command()
def chips(
    split: list[str] = typer.Option(DEFAULT_SPLITS, help="Split files whose chips to process."),
    workers: int = typer.Option(4, help="Parallel chips (each streams its own tiles)."),
    cache_dir: Path = typer.Option(None, help="Keep whole tiles here instead of streaming."),
    limit: int = typer.Option(0, help="Only the first N chips (0 = all)."),
) -> None:
    """HAND and slope for every chip in the split files whose S1 chip exists locally."""
    rows = {}
    for name in split:
        for row in read_split(name).itertuples():
            rows[row.chip_id] = row
    jobs = [r for r in rows.values() if data_path(r.s1).exists()]
    missing = len(rows) - len(jobs)
    if missing:
        log.warning("%d chips skipped: S1 file not downloaded", missing)
    if limit:
        jobs = jobs[:limit]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                build_one, data_path(r.s1), data_path(r.hand), data_path(r.slope), cache_dir
            )
            for r in jobs
        ]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                log.info("%d / %d chips done", done, len(jobs))


@app.command()
def raster(
    path: Path = typer.Argument(..., help="Any GeoTIFF; HAND and slope use its grid."),
    out_dir: Path = typer.Option(None, help="Default: next to the input file."),
    block: int = typer.Option(2048, help="Block size in pixels for large scenes."),
    cache_dir: Path = typer.Option(None, help="Keep whole tiles here instead of streaming."),
) -> None:
    """Write ``<stem>_HAND.tif`` and ``<stem>_SLOPE.tif`` on the grid of ``path``."""
    outputs = write_terrain_rasters(path, out_dir or path.parent, block, cache_dir)
    for out in outputs.values():
        log.info("wrote %s", out)


@app.command()
def figure(
    chip_ids: list[str] = typer.Argument(..., help="Chip ids, e.g. Bolivia_103757."),
    out: Path = typer.Option(None, help="Default: docs/figures/hand_vs_label.png"),
) -> None:
    """PNG with VH (dB), the hand label and HAND for each chip, one row per chip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or repo_root() / "docs" / "figures" / "hand_vs_label.png"
    table = {}
    for name in ("s11_train", "s11_val", "s11_test", "s11_bolivia"):
        for row in read_split(name).itertuples():
            table[row.chip_id] = row
    fig, axes = plt.subplots(len(chip_ids), 3, figsize=(10, 3.4 * len(chip_ids)), squeeze=False)
    for ax_row, chip_id in zip(axes, chip_ids, strict=True):
        row = table[chip_id]
        vh = read_raster(data_path(row.s1)).array[1]
        label = read_raster(data_path(row.label)).array[0].astype(float)
        label[label < 0] = np.nan
        hand = read_raster(data_path(row.hand)).array[0]
        panels = [
            (vh, "VH (dB)", "gray", (-30, -5)),
            (label, "hand label (water = 1)", "Blues", (0, 1)),
            (hand, "HAND (m)", "viridis_r", (0, 20)),
        ]
        for ax, (img, title, cmap, (lo, hi)) in zip(ax_row, panels, strict=True):
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{chip_id}: {title}", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    log.info("wrote %s", out)


if __name__ == "__main__":
    app()
