"""Sen1Floods11 v1.1 layout, split files and chip paths.

Facts checked on real chips (``scripts/inspect_sen1floods11.py``):

* ``S1Hand`` / ``S1Weak``: 2 bands float32 in dB (band 1 VV, band 2 VH), 512x512, EPSG:4326,
  pixel 8.98e-5 degrees (about 10 m), nodata NaN. Values roughly -50 to +25 dB.
* ``LabelHand``: int16, -1 no data, 0 not water, 1 water (no nodata tag in the file).
* ``JRCWaterHand``: uint8 0/1 permanent water. ``S1OtsuLabelHand``: int16 0/1, nodata -1.
* 446 hand-labelled chips in 11 events; 4,384 weak S1 chips in 12 events (adds Colombia,
  includes 224 Bolivia chips, which must not be used when Bolivia is a test event).
* The official train/valid/test splits are random **within** events; only Bolivia is a held-out
  event. ``s11_southasia_holdout_*`` adds a second held-out region (India, Pakistan, Sri Lanka).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import data_path, repo_root

ROOT = "sen1floods11/v1.1"
HAND_DIR = f"{ROOT}/data/flood_events/HandLabeled"
WEAK_DIR = f"{ROOT}/data/flood_events/WeaklyLabeled"
DERIVED_DIR = "sen1floods11"  # HAND/ and SLOPE/ built by scripts/prepare_hand.py

SOUTH_ASIA = ("India", "Pakistan", "Sri-Lanka")
OFFICIAL_SPLITS = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
BANDS = ("VV", "VH")


def event_of(chip_id: str) -> str:
    """Event (country or region) of a chip id such as ``Sri-Lanka_120804`` -> ``Sri-Lanka``."""
    return chip_id.split("_")[0]


def hand_labelled_paths(chip_id: str) -> dict[str, str]:
    """Paths (relative to ``$DATA_ROOT``) of every file of one hand-labelled chip."""
    return {
        "s1": f"{HAND_DIR}/S1Hand/{chip_id}_S1Hand.tif",
        "label": f"{HAND_DIR}/LabelHand/{chip_id}_LabelHand.tif",
        "jrc": f"{HAND_DIR}/JRCWaterHand/{chip_id}_JRCWaterHand.tif",
        "otsu": f"{HAND_DIR}/S1OtsuLabelHand/{chip_id}_S1OtsuLabelHand.tif",
        "hand": f"{DERIVED_DIR}/HAND/{chip_id}_HAND.tif",
        "slope": f"{DERIVED_DIR}/SLOPE/{chip_id}_SLOPE.tif",
    }


def weak_paths(chip_id: str) -> dict[str, str]:
    """Paths (relative to ``$DATA_ROOT``) of one weakly labelled chip."""
    return {
        "s1": f"{WEAK_DIR}/S1Weak/{chip_id}_S1Weak.tif",
        "otsu": f"{WEAK_DIR}/S1OtsuLabelWeak/{chip_id}_S1OtsuLabelWeak.tif",
        "hand": f"{DERIVED_DIR}/HAND/{chip_id}_HAND.tif",
        "slope": f"{DERIVED_DIR}/SLOPE/{chip_id}_SLOPE.tif",
    }


def read_split(name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    """Read ``splits/<name>.csv`` (e.g. ``s11_train``). Split files are never generated."""
    path = (splits_dir or repo_root() / "splits") / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return pd.read_csv(path)


def resolve(relative: str) -> Path:
    """Absolute path of a split-file entry."""
    return data_path(relative)
