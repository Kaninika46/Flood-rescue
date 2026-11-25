"""Download and prepare SpaceNet 8 (prompt B1). Run in the "sn8" environment.

    python scripts/download_sn8.py download     # 4.7 GB: Germany + Louisiana-East tarballs
    python scripts/download_sn8.py prepare      # baseline geojson_prep + create_masks
    python scripts/download_sn8.py check        # tiles and masks per area vs the split files

Sizes (bucket listing, November 2026): Germany_Training_Public.tar.gz 1.36 GB,
Louisiana-East_Training_Public.tar.gz 3.34 GB. Louisiana-West (test) has no public labels and
is not downloaded. Data lands in ``$DATA_ROOT/sn8/<area>/``.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, repo_root
from src.common.logging import get_logger
from src.level2_roads.data import SN8_DIR, TRAIN_AOIS, read_sn8_split

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods"
log = get_logger("sn8")
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def download(
    keep_tarballs: bool = typer.Option(False, help="Keep the .tar.gz after extracting."),
) -> None:
    """Fetch and extract the two training areas plus resolutions.txt."""
    root = data_path(SN8_DIR)
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET}/resolutions.txt", root / "resolutions.txt")
    for aoi in TRAIN_AOIS:
        if (root / aoi / "PRE-event").is_dir():
            log.info("%s already extracted", aoi)
            continue
        archive = root / f"{aoi}.tar.gz"
        if not archive.exists():
            log.info("downloading %s", archive.name)
            urllib.request.urlretrieve(f"{BUCKET}/tarballs/{aoi}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        if not keep_tarballs:
            archive.unlink()
        log.info("extracted %s", aoi)


@app.command()
def prepare() -> None:
    """Run the baseline data preparation (our verbatim copies) on both areas."""
    root = str(data_path(SN8_DIR))
    prep = repo_root() / "src" / "level2_roads" / "sn8" / "data_prep"
    for script in ("geojson_prep.py", "create_masks.py"):
        cmd = [sys.executable, str(prep / script), "--root_dir", root, "--aoi_dirs", *TRAIN_AOIS]
        log.info("running %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=prep)


@app.command()
def check() -> None:
    """Count images and masks per area and confirm every split row exists."""
    for aoi in TRAIN_AOIS:
        base = data_path(SN8_DIR, aoi)
        counts = {d: len(list((base / d).glob("*.tif"))) for d in ("PRE-event", "POST-event")}
        counts["masks"] = len(list((base / "annotations" / "masks").glob("*.tif")))
        counts["geojson"] = len(list((base / "annotations").glob("*.geojson")))
        print(f"{aoi}: {counts}")
    for name in ("sn8_train", "sn8_val", "sn8_cross_train", "sn8_cross_test"):
        df = read_sn8_split(name)
        missing = sum(
            not data_path(p).exists() for col in ("preimg", "postimg", "flood") for p in df[col]
        )
        print(f"{name}: {len(df)} tiles, {missing} missing files")


if __name__ == "__main__":
    app()
