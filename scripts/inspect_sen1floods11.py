"""Print shape, dtype, CRS, nodata, value ranges and label counts for a few Sen1Floods11 chips.

python scripts/inspect_sen1floods11.py                    # 2 chips from each split
python scripts/inspect_sen1floods11.py --per-split 5 --split s11_bolivia
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import typer

from src.common.config import data_path
from src.common.raster import describe_raster, read_raster
from src.level1_radar.s11 import read_split

app = typer.Typer(add_completion=False, help=__doc__)


def _counts(path: Path) -> str:
    array = read_raster(path).array
    if array.dtype.kind == "f":
        return f"NaN pixels: {int(np.isnan(array).sum())}"
    values, counts = np.unique(array, return_counts=True)
    return "values " + ", ".join(f"{v}: {c}" for v, c in zip(values, counts, strict=True))


@app.command()
def main(
    split: list[str] = typer.Option(
        ["s11_train", "s11_val", "s11_test", "s11_bolivia"], help="Split files to sample."
    ),
    per_split: int = typer.Option(2, help="Chips per split."),
) -> None:
    """Describe the first chips of each split."""
    for name in split:
        df = read_split(name).head(per_split)
        print(f"\n### {name} ({len(read_split(name))} chips)")
        for row in df.itertuples():
            print(f"- {row.chip_id} (event {row.event})")
            for column in ("s1", "label", "jrc", "otsu"):
                path = data_path(getattr(row, column))
                if not path.exists():
                    print(f"    {column:5s} MISSING {path}")
                    continue
                info = describe_raster(path)
                ranges = [
                    (round(lo, 2), round(hi, 2)) if lo is not None else None
                    for lo, hi in info["band_ranges"]
                ]
                print(
                    f"    {column:5s} {info['shape']} {info['dtype']} {info['crs']} "
                    f"nodata={info['nodata']} ranges={ranges} {_counts(path)}"
                )


if __name__ == "__main__":
    app()
