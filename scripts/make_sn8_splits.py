"""Write the committed SpaceNet 8 split files (prompt B1). Run once.

Inputs: ``$DATA_ROOT/sn8/<area>/<area>_label_image_mapping.csv`` for the two public training
areas (``scripts/download_sn8.py`` fetches them first).

* ``sn8_train`` / ``sn8_val``: the baseline's random 85/15 tile split over Germany and
  Louisiana-East (S0). The shuffle is fixed (seed 0) and written once, never redone in training.
  Tiles of one area are spatially correlated, so S0 is optimistic; S1 is the honest test.
* ``sn8_cross_train`` (Louisiana-East) / ``sn8_cross_test`` (Germany): split by area (S1).

Columns: aoi, event, tile_id, preimg, postimg, postimg2, label, building, road, roadspeed, flood
(paths relative to ``$DATA_ROOT``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import typer

from src.common.config import repo_root
from src.level2_roads.data import TRAIN_AOIS, rows_from_mapping

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(val_percent: float = typer.Option(0.15), seed: int = typer.Option(0)) -> None:
    """Build and write the four split files and print counts per area."""
    by_aoi = {aoi: pd.DataFrame(rows_from_mapping(aoi)) for aoi in TRAIN_AOIS}
    both = pd.concat(by_aoi.values(), ignore_index=True)
    order = np.random.default_rng(seed).permutation(len(both))
    n_val = math.ceil(len(both) * val_percent)
    files = {
        "sn8_val": both.iloc[order[:n_val]],
        "sn8_train": both.iloc[order[n_val:]],
        "sn8_cross_train": by_aoi["Louisiana-East_Training_Public"],
        "sn8_cross_test": by_aoi["Germany_Training_Public"],
    }
    out = repo_root() / "splits"
    for name, df in files.items():
        df = df.sort_values(["aoi", "tile_id"]).reset_index(drop=True)
        df.to_csv(out / f"{name}.csv", index=False)
        counts = df["aoi"].value_counts().to_dict()
        print(f"{name:16s} {len(df):4d} tiles  {counts}")


if __name__ == "__main__":
    app()
