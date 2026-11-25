"""Pseudo-labels for the weakly labelled Sen1Floods11 chips (E5 / E6, prompt A7).

Idea from ``third_party/etci2021/notebooks/Generate_Pseudo.ipynb`` (Apache-2.0): run an
ensemble on unlabelled images and keep only what it is sure about. ETCI kept whole tiles where
90% of pixels were confident; we decide **per pixel**:

* water (1) where the ensemble mean is above ``high`` and (optionally) every member says water
* not water (0) where the mean is below ``low`` and every member says not water
* ignore (-1) everywhere else, and where the radar has no data

    python -m src.level1_radar.pseudo_label --config configs/E5.yaml

Writes ``$DATA_ROOT/<data.pseudo.dir>/<chip>_pseudo.tif`` (int16, nodata -1) on each chip's grid,
plus ``summary.csv`` with the kept fraction per chip; the totals are logged to the tracker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.common.config import data_path, load_config
from src.common.logging import get_logger, start_run
from src.common.raster import read_raster, write_raster
from src.level1_radar.predict import Ensemble
from src.level1_radar.s11 import read_split

log = get_logger("pseudo_label")
app = typer.Typer(add_completion=False, help=__doc__)


def pseudo_from_probs(
    probs: np.ndarray, high: float, low: float, require_agreement: bool = True
) -> np.ndarray:
    """Pseudo-label ``(H, W)`` int16 from member probabilities ``(M, H, W)``."""
    mean = probs.mean(axis=0)
    out = np.full(mean.shape, -1, np.int16)
    water = mean > high
    land = mean < low
    if require_agreement:
        water &= (probs > 0.5).all(axis=0)
        land &= (probs < 0.5).all(axis=0)
    out[water] = 1
    out[land] = 0
    out[np.isnan(probs).any(axis=0)] = -1
    return out


def member_probs(ensemble: Ensemble, layers: dict[str, np.ndarray]) -> np.ndarray:
    """Water probability of every member, ``(M, H, W)``."""
    out = []
    for member in ensemble.members:
        image = np.stack([layers[c] for c in member.channels]).astype(np.float32)
        out.append(ensemble._predict_member(member, member.norm.apply(image, member.channels)))
    return np.stack(out)


@app.command()
def main(
    config: Path = typer.Option(..., help="configs/E5.yaml (pseudo_label + data.pseudo sections)"),
    set_: list[str] = typer.Option([], "--set", help="Config override key.sub=value."),
    limit: int = typer.Option(0, help="Only the first N weak chips (0 = all)."),
    overwrite: bool = typer.Option(False, help="Recompute existing pseudo-label files."),
) -> None:
    """Write pseudo-labels for every weak chip not in an excluded event."""
    cfg = load_config(config, set_)
    plc, pseudo = cfg["pseudo_label"], cfg["data"]["pseudo"]
    paths = [
        data_path("checkpoints", f"{plc['from_experiment']}-s{s}", "best.pt") for s in plc["seeds"]
    ]
    ensemble = Ensemble.from_checkpoints(paths)
    rows = read_split(pseudo.get("split", "s11_weak"))
    rows = rows[~rows["event"].isin(pseudo.get("exclude_events", []))]
    rows = rows[[data_path(p).exists() for p in rows["s1"]]]
    if limit:
        rows = rows.head(limit)
    out_dir = data_path(pseudo["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    with start_run(cfg, 0, job_type="pseudo_label") as run:
        for i, row in enumerate(rows.to_dict("records"), start=1):
            out = out_dir / f"{row['chip_id']}_pseudo.tif"
            if out.exists() and not overwrite:
                label = read_raster(out).array[0]
            else:
                probs = member_probs(ensemble, ensemble.load_layers(row))
                label = pseudo_from_probs(
                    probs, plc["high"], plc["low"], plc.get("require_agreement", True)
                )
                grid = read_raster(data_path(row["s1"]), bands=1)
                write_raster(out, label, crs=grid.crs, transform=grid.transform, nodata=-1)
            stats.append(
                {
                    "chip_id": row["chip_id"],
                    "event": row["event"],
                    "kept": float((label >= 0).mean()),
                    "water": float((label == 1).mean()),
                }
            )
            if i % 200 == 0:
                log.info("%d / %d chips", i, len(rows))
        df = pd.DataFrame(stats)
        df.to_csv(out_dir / "summary.csv", index=False)
        total = {
            "pseudo/chips": len(df),
            "pseudo/kept_fraction": float(df["kept"].mean()),
            "pseudo/water_fraction_of_all": float(df["water"].mean()),
        }
        run.log_summary(total)
        run.log_file(out_dir / "summary.csv")
    log.info("pseudo-labels in %s: %s", out_dir, total)


if __name__ == "__main__":
    app()
