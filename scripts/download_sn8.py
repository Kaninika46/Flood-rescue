"""Download and prepare SpaceNet 8 (prompt B1). Run in the "sn8" environment.

    python scripts/download_sn8.py download     # 4.7 GB: Germany + Louisiana-East tarballs
    python scripts/download_sn8.py prepare      # baseline geojson_prep + create_masks
    python scripts/download_sn8.py check        # tiles and masks per area vs the split files

Sizes (bucket listing, November 2026): Germany_Training_Public.tar.gz 1.36 GB,
Louisiana-East_Training_Public.tar.gz 3.34 GB. Louisiana-West (test) has no public labels and
is not downloaded. Data lands in ``$DATA_ROOT/sn8/<area>/``.
"""

from __future__ import annotations

import http.client
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, repo_root
from src.common.logging import get_logger
from src.level2_roads.data import MASK_KINDS, SN8_DIR, TRAIN_AOIS, read_sn8_split

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods"
log = get_logger("sn8")
app = typer.Typer(add_completion=False, help=__doc__)


def download_archive(url: str, archive: Path) -> None:
    """Resume interrupted downloads and keep incomplete bytes out of completed archives."""
    partial = archive.with_name(f"{archive.name}.part")
    mib = 1024**2
    attempts = 5
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status == 206:
                    content_range = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if not content_range or int(content_range[1]) != offset:
                        raise OSError("server returned an unexpected Content-Range")
                    expected = int(content_range[3])
                    log.info("resuming %s at %.2f MiB", archive.name, offset / mib)
                else:
                    offset = 0
                    expected = int(response.headers.get("Content-Length", "0"))

                received = last_bytes = offset
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(mib):
                        output.write(block)
                        received += len(block)
                        now = time.monotonic()
                        elapsed = now - last_report
                        if elapsed < 10 and received != expected:
                            continue
                        rate = (received - last_bytes) / elapsed
                        if expected and rate > 0:
                            remaining = max(0, expected - received)
                            log.info(
                                "%s: %.2f/%.2f MiB (%.1f%%), %.2f MiB left, %.2f MiB/s, ETA ~%.0fs",
                                archive.name,
                                received / mib,
                                expected / mib,
                                100 * received / expected,
                                remaining / mib,
                                rate / mib,
                                remaining / rate,
                            )
                        else:
                            log.info("%s: %.2f MiB received", archive.name, received / mib)
                        last_report, last_bytes = now, received
                if expected and received != expected:
                    raise OSError(f"short transfer: {received} of {expected} bytes")
        except urllib.error.HTTPError as error:
            if (
                offset
                and error.code == 416
                and error.headers.get("Content-Range") == f"bytes */{offset}"
            ):
                partial.replace(archive)
                return
            if attempt == attempts:
                raise
            log.warning("%s: attempt %d/%d failed: %s", archive.name, attempt, attempts, error)
            time.sleep(2)
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            log.warning("%s: attempt %d/%d failed: %s", archive.name, attempt, attempts, error)
            time.sleep(2)
        else:
            partial.replace(archive)
            return


@app.command()
def download(
    keep_tarballs: bool = typer.Option(False, help="Keep the .tar.gz after extracting."),
) -> None:
    """Fetch and extract the two training areas plus resolutions.txt."""
    root = data_path(SN8_DIR)
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET}/resolutions.txt", root / "resolutions.txt")
    for aoi in TRAIN_AOIS:
        destination = root / aoi
        required_dirs = ("PRE-event", "POST-event", "annotations")
        if all((destination / name).is_dir() for name in required_dirs):
            log.info("%s already extracted", aoi)
            continue
        archive = root / f"{aoi}.tar.gz"
        if not archive.is_file() or archive.stat().st_size == 0:
            log.info("downloading %s", archive.name)
            download_archive(f"{BUCKET}/tarballs/{aoi}.tar.gz", archive)
        log.info("extracting %s", archive.name)
        with tempfile.TemporaryDirectory(prefix=f".{aoi}-", dir=root) as staging:
            extracted = Path(staging) / "area"
            extracted.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter="data")
            if (extracted / aoi).is_dir():
                extracted /= aoi
            if not all((extracted / name).is_dir() for name in required_dirs):
                raise ValueError(f"{archive.name} does not contain the expected SpaceNet 8 folders")
            extracted.rename(destination)
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
    required_columns = ("preimg", "postimg", "label", *MASK_KINDS)
    failed = False
    for name in ("sn8_train", "sn8_val", "sn8_cross_train", "sn8_cross_test"):
        df = read_sn8_split(name)
        missing = [p for col in required_columns for p in df[col] if not data_path(p).is_file()]
        print(f"{name}: {len(df)} tiles, {len(missing)} missing required files")
        if missing:
            failed = True
            for path in missing[:5]:
                print(f"  missing: {path}")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
