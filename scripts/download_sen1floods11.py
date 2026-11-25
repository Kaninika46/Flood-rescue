"""Download the parts of Sen1Floods11 v1.1 this project uses into ``$DATA_ROOT/sen1floods11``.

The whole bucket is 34 GiB (mostly Sentinel-2 and permanent-water chips we do not use). Parts:

    hand   hand-labelled S1, labels, JRC water, Otsu labels   about 0.7 GB
    weak   weakly labelled S1 chips + their Otsu labels        about 9.5 GB
    meta   official split CSVs and the metadata GeoJSON        < 1 MB

    python scripts/download_sen1floods11.py                   # meta + hand
    python scripts/download_sen1floods11.py --part weak       # needed for E5/E6 pseudo-labels

Needs ``gsutil`` (in the "flood" env). Re-running only fetches missing files (``rsync``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import get_data_root

BUCKET = "gs://sen1floods11/v1.1"
PARTS: dict[str, list[str]] = {
    "meta": ["splits/flood_handlabeled"],
    "hand": [
        "data/flood_events/HandLabeled/S1Hand",
        "data/flood_events/HandLabeled/LabelHand",
        "data/flood_events/HandLabeled/JRCWaterHand",
        "data/flood_events/HandLabeled/S1OtsuLabelHand",
    ],
    "weak": [
        "data/flood_events/WeaklyLabeled/S1Weak",
        "data/flood_events/WeaklyLabeled/S1OtsuLabelWeak",
    ],
}
SIZES_GB = {"meta": 0.001, "hand": 0.7, "weak": 9.5}

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    part: list[str] = typer.Option(["meta", "hand"], help="Parts to fetch: meta, hand, weak."),
    dry_run: bool = typer.Option(False, help="Only print the commands."),
) -> None:
    """Fetch the chosen parts with ``gsutil -m rsync``."""
    unknown = set(part) - set(PARTS)
    if unknown:
        raise typer.BadParameter(f"unknown part(s) {sorted(unknown)}; choose from {list(PARTS)}")
    if shutil.which("gsutil") is None:
        raise typer.Exit("gsutil not found: activate the 'flood' conda env")
    root = get_data_root() / "sen1floods11" / "v1.1"
    print(f"Target: {root}  (about {sum(SIZES_GB[p] for p in part):.1f} GB)")
    for name in part:
        for sub in PARTS[name]:
            dest = root / sub
            dest.mkdir(parents=True, exist_ok=True)
            cmd = ["gsutil", "-m", "-q", "rsync", "-r", f"{BUCKET}/{sub}", str(dest)]
            print(" ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True)
    if "meta" in part and not dry_run:
        meta = root / "Sen1Floods11_Metadata.geojson"
        subprocess.run(
            ["gsutil", "-q", "cp", f"{BUCKET}/Sen1Floods11_Metadata.geojson", str(meta)],
            check=True,
        )
    print("Done.")


if __name__ == "__main__":
    app()
