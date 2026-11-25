"""Compute per-channel mean and standard deviation on ``s11_train`` only (prompt A5).

    python scripts/compute_s11_norm.py            # writes configs/s11_norm.yaml

The South Asia hold-out experiments use the same statistics: they are a property of the input
units (dB, metres, degrees), and recomputing them per split would make runs harder to compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
import yaml

from src.common.config import repo_root
from src.level1_radar.datasets import ALL_CHANNELS, compute_norm
from src.level1_radar.s11 import read_split

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    split: str = typer.Option("s11_train", help="Split to compute the statistics on."),
    out: Path = typer.Option(None, help="Default: configs/s11_norm.yaml"),
) -> None:
    """Write the statistics YAML."""
    out = out or repo_root() / "configs" / "s11_norm.yaml"
    norm = compute_norm(read_split(split), list(ALL_CHANNELS))
    data = {
        "computed_on": split,
        "units": {"VV": "dB", "VH": "dB", "HAND": "m", "SLOPE": "degrees"},
        "mean": {k: round(v, 4) for k, v in norm.mean.items()},
        "std": {k: round(v, 4) for k, v in norm.std.items()},
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    app()
