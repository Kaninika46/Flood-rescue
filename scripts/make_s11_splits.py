"""Write the committed Sen1Floods11 split files in ``splits/`` (prompt A1). Run once.

Inputs: the official split CSVs (``$DATA_ROOT/sen1floods11/v1.1/splits/flood_handlabeled``)
and a listing of the weak chips (local folder if downloaded, else ``gsutil ls``).

Outputs (paths relative to ``$DATA_ROOT``):

* ``s11_train``, ``s11_val``, ``s11_test``, ``s11_bolivia``: the official hand-labelled splits.
* ``s11_southasia_holdout_{train,val,test}``: every India, Pakistan and Sri Lanka chip moves to
  test; train/val are the official ones without those events (Bolivia stays out entirely).
* ``s11_weak``: all weak chips with their event, for pseudo-labelling (filter events per config).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import typer

from src.common.config import data_path, repo_root
from src.level1_radar.s11 import (
    OFFICIAL_SPLITS,
    ROOT,
    SOUTH_ASIA,
    WEAK_DIR,
    event_of,
    hand_labelled_paths,
    weak_paths,
)

app = typer.Typer(add_completion=False, help=__doc__)


def _official(name: str) -> pd.DataFrame:
    csv = data_path(ROOT, "splits", "flood_handlabeled", OFFICIAL_SPLITS[name], must_exist=True)
    raw = pd.read_csv(csv, header=None, names=["s1_file", "label_file"])
    rows = []
    for s1_file in raw["s1_file"]:
        chip_id = s1_file.removesuffix("_S1Hand.tif")
        rows.append(
            {"chip_id": chip_id, "event": event_of(chip_id), **hand_labelled_paths(chip_id)}
        )
    return pd.DataFrame(rows)


def _weak_ids() -> list[str]:
    local = data_path(WEAK_DIR, "S1Weak")
    if local.is_dir():
        names = [p.name for p in local.glob("*_S1Weak.tif")]
    else:
        out = subprocess.run(
            ["gsutil", "ls", "gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/"],
            check=True,
            capture_output=True,
            text=True,
        )
        names = [line.rsplit("/", 1)[-1] for line in out.stdout.split()]
    return sorted(n.removesuffix("_S1Weak.tif") for n in names if n.endswith("_S1Weak.tif"))


@app.command()
def main(out_dir: Path = typer.Option(None, help="Default: splits/ in the repo.")) -> None:
    """Build and write every Sen1Floods11 split file."""
    out_dir = out_dir or repo_root() / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    official = {name: _official(name) for name in OFFICIAL_SPLITS}
    files: dict[str, pd.DataFrame] = {f"s11_{n}": df for n, df in official.items()}

    in_sa = {n: df["event"].isin(SOUTH_ASIA) for n, df in official.items()}
    files["s11_southasia_holdout_train"] = official["train"][~in_sa["train"]]
    files["s11_southasia_holdout_val"] = official["val"][~in_sa["val"]]
    files["s11_southasia_holdout_test"] = pd.concat(
        [official[n][in_sa[n]] for n in ("train", "val", "test")], ignore_index=True
    )

    weak = [{"chip_id": c, "event": event_of(c), **weak_paths(c)} for c in _weak_ids()]
    files["s11_weak"] = pd.DataFrame(weak)

    for name, df in files.items():
        df = df.sort_values(["event", "chip_id"]).reset_index(drop=True)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        counts = ", ".join(f"{e} {c}" for e, c in df["event"].value_counts().sort_index().items())
        print(f"{name:32s} {len(df):5d} chips  ({counts})")


if __name__ == "__main__":
    app()
