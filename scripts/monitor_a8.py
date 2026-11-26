"""Follow the A8 pipeline and measure completed Sen1Floods11 download progress."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

from download_sen1floods11 import BUCKET, PARTS

STAGE_EVENT = re.compile(r"^\[[^]]+\] (Starting|Completed) (\S+)")
DOWNLOAD_PARTS = {"hand-download": ("meta", "hand"), "weak-download": ("weak",)}
MIB = 1024**2
GIB = 1024**3


def download_manifest(stage: str, data_root: Path) -> dict[Path, int]:
    bucket = urlsplit(BUCKET)
    version = bucket.path.strip("/")
    parts = DOWNLOAD_PARTS[stage]
    targets = [f"{version}/{folder}/" for part in parts for folder in PARTS[part]]
    if "meta" in parts:
        targets.append(f"{version}/Sen1Floods11_Metadata.geojson")

    objects = {}
    endpoint = f"https://storage.googleapis.com/storage/v1/b/{bucket.netloc}/o"
    for prefix in targets:
        query = {"prefix": prefix, "fields": "nextPageToken,items(name,size)"}
        while True:
            with urlopen(f"{endpoint}?{urlencode(query)}", timeout=10) as response:
                page = json.load(response)
            for item in page.get("items", ()):
                name = item["name"]
                if name.endswith("/") or (not prefix.endswith("/") and name != prefix):
                    continue
                relative = Path(name)
                if not relative.is_absolute() and ".." not in relative.parts:
                    objects[data_root / bucket.netloc / relative] = int(item["size"])
            if "nextPageToken" not in page:
                break
            query["pageToken"] = page["nextPageToken"]
    if not objects:
        raise ValueError("the public bucket returned an empty object list")
    return objects


def completed_files(manifest: dict[Path, int]) -> tuple[int, int]:
    count = downloaded = 0
    for path, expected in manifest.items():
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size == expected:
            count += 1
            downloaded += size
    return count, downloaded


def format_eta(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02}m" if hours else f"{minutes}m {seconds:02}s"


def runner_active(lock: Path) -> bool:
    try:
        with lock.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except FileNotFoundError:
        return False
    except BlockingIOError:
        return True
    return False


def watch(data_root: Path, lines: int, interval: float) -> None:
    run_dir = data_root / "a8"
    log = run_dir / "run.log"
    print(f"Following {log}  |  Ctrl+C closes this view")
    print("Transfer speed counts completed files using a rolling sample of about 30s.")
    if not log.is_file():
        raise SystemExit(f"Run log does not exist yet: {log}")

    stage = None
    history: deque[str] = deque(maxlen=lines)
    samples: deque[tuple[float, int]] = deque()
    manifest = pending = None
    total = 0
    next_update = 0.0
    active_before = None
    with (
        log.open(encoding="utf-8", errors="replace") as output,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        for line in output:
            history.append(
                line.rstrip().split("  (about ", 1)[0]
                if line.startswith("Target: ")
                else line.rstrip()
            )
            if event := STAGE_EVENT.match(line):
                stage = event[2] if event[1] == "Starting" else None
        for line in history:
            print(line)
        if stage:
            print(f"Current stage: {stage}")

        while True:
            for line in output:
                print(
                    line.rstrip().split("  (about ", 1)[0]
                    if line.startswith("Target: ")
                    else line.rstrip()
                )
                if event := STAGE_EVENT.match(line):
                    stage = event[2] if event[1] == "Starting" else None
                    if pending is not None:
                        pending.cancel()
                    pending = manifest = None
                    samples.clear()
                    next_update = 0.0

            active = runner_active(run_dir / "run.lock")
            if active != active_before:
                print("Runner active." if active else "Runner idle. Waiting for new output.")
                active_before = active

            now = time.monotonic()
            if not active or stage not in DOWNLOAD_PARTS or now < next_update:
                time.sleep(0.5)
                continue

            if manifest is None:
                if pending is None:
                    print(f"{stage}: reading exact object sizes from the public bucket...")
                    pending = executor.submit(download_manifest, stage, data_root)
                if not pending.done():
                    time.sleep(0.5)
                    continue
                try:
                    manifest = pending.result()
                except (OSError, ValueError) as error:
                    print(f"{stage}: bucket sizes unavailable ({error}). Retrying in 30s.")
                    pending = None
                    next_update = now + 30
                    continue
                total = sum(manifest.values())

            files, downloaded = completed_files(manifest)
            samples.append((now, downloaded))
            while len(samples) > 2 and samples[1][0] <= now - 30:
                samples.popleft()
            first_time, first_bytes = samples[0]
            elapsed = now - first_time
            rate = max(0.0, (downloaded - first_bytes) / elapsed) if elapsed else None
            remaining = total - downloaded
            speed = "measuring..." if rate is None else f"{rate / MIB:.2f} MiB/s"
            eta = f"~{format_eta(remaining / rate)}" if rate else "--"
            print(
                f"{stage}: {downloaded / GIB:.2f}/{total / GIB:.2f} GiB "
                f"({downloaded / total:.1%}, {files:,}/{len(manifest):,} files) | "
                f"left {remaining / GIB:.2f} GiB | {speed} | ETA {eta}"
            )
            next_update = now + interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path.home() / "flood-data")),
    )
    parser.add_argument("--lines", type=int, default=20, help="Recent log lines to show on attach.")
    parser.add_argument(
        "--interval", type=float, default=5, help="Seconds between download samples."
    )
    args = parser.parse_args()
    if args.lines < 0 or args.interval < 1:
        parser.error("--lines must be nonnegative and --interval must be at least 1 second")
    try:
        watch(args.data_root, args.lines, args.interval)
    except KeyboardInterrupt:
        print("\nClosed the monitoring view.")


if __name__ == "__main__":
    main()
