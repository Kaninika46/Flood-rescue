"""Sen1Floods11 chips as tensors for training and evaluation.

Channels are chosen by name: ``VV`` and ``VH`` (dB, from the S1 chip), ``HAND`` (m) and
``SLOPE`` (degrees) from ``scripts/prepare_hand.py``. Each channel is standardised with the
mean and standard deviation of ``s11_train`` (``configs/s11_norm.yaml``); missing values
(NaN) become 0, i.e. the training mean. Labels keep -1 as "ignore".

Training items are random ``crop x crop`` windows with random flips and 90-degree rotations;
evaluation items are full 512x512 chips. Augmentation uses the torch RNG, so a DataLoader
with ``generator=torch_generator(seed)`` and ``worker_init_fn=seed_worker`` is reproducible.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from src.common.config import data_path
from src.common.raster import read_raster

S1_CHANNELS = ("VV", "VH")
TERRAIN_CHANNELS = ("HAND", "SLOPE")
ALL_CHANNELS = S1_CHANNELS + TERRAIN_CHANNELS


@dataclass(frozen=True)
class Norm:
    """Per-channel mean and standard deviation (computed on ``s11_train`` only)."""

    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> Norm:
        """Load ``configs/s11_norm.yaml``."""
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(mean=dict(data["mean"]), std=dict(data["std"]))

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Plain dict (stored inside checkpoints)."""
        return {"mean": dict(self.mean), "std": dict(self.std)}

    def apply(self, image: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        """Standardise a ``(C, H, W)`` image in physical units; NaN becomes 0."""
        out = np.empty_like(image, dtype=np.float32)
        for i, name in enumerate(channels):
            out[i] = (image[i] - self.mean[name]) / self.std[name]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _check_channels(channels: Sequence[str]) -> None:
    unknown = [c for c in channels if c not in ALL_CHANNELS]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; choose from {ALL_CHANNELS}")


def load_image(paths: dict[str, Any], channels: Sequence[str]) -> np.ndarray:
    """Stack the requested channels of one chip, in physical units, as ``(C, H, W)`` float32.

    Args:
        paths: Mapping with ``s1`` and, when needed, ``hand`` / ``slope`` paths relative to
            ``$DATA_ROOT`` (a split-file row works).
        channels: Channel names, e.g. ``["VV", "VH", "HAND"]``.
    """
    _check_channels(channels)
    layers: list[np.ndarray] = []
    s1 = None
    for name in channels:
        if name in S1_CHANNELS:
            if s1 is None:
                s1 = read_raster(data_path(paths["s1"])).array.astype(np.float32)
            layers.append(s1[S1_CHANNELS.index(name)])
        else:
            path = data_path(paths[name.lower()])
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run scripts/prepare_hand.py first")
            layers.append(read_raster(path).array[0].astype(np.float32))
    return np.stack(layers)


def load_label(path: str) -> np.ndarray:
    """Label raster (``-1`` ignore, 0, 1) as int16 ``(H, W)``."""
    return read_raster(data_path(path)).array[0].astype(np.int16)


def random_crop_box(
    label: np.ndarray, crop: int, generator: torch.Generator | None, tries: int = 10
) -> tuple[int, int]:
    """Top-left corner of a random crop; prefers crops with at least 1% labelled pixels."""
    height, width = label.shape
    best = (0, 0)
    best_valid = -1.0
    for _ in range(tries):
        top = int(torch.randint(0, height - crop + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - crop + 1, (1,), generator=generator))
        valid = float((label[top : top + crop, left : left + crop] >= 0).mean())
        if valid >= 0.01:
            return top, left
        if valid > best_valid:
            best, best_valid = (top, left), valid
    return best


def augment(
    image: np.ndarray, label: np.ndarray, generator: torch.Generator | None
) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal/vertical flips and 90-degree rotation (the D4 group)."""
    if bool(torch.randint(0, 2, (1,), generator=generator)):
        image, label = image[:, :, ::-1], label[:, ::-1]
    if bool(torch.randint(0, 2, (1,), generator=generator)):
        image, label = image[:, ::-1, :], label[::-1, :]
    k = int(torch.randint(0, 4, (1,), generator=generator))
    if k:
        image, label = np.rot90(image, k, axes=(1, 2)), np.rot90(label, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class Sen1Floods11Dataset(Dataset):
    """Chips listed in a split DataFrame.

    Args:
        rows: Split rows (``chip_id``, ``event``, ``s1`` and the label/terrain columns).
        channels: Input channels by name.
        norm: Normalisation statistics.
        train: Random crops and augmentation when True; full chips otherwise.
        crop: Training crop size.
        label_column: Column holding the label path (``label`` for hand labels, ``target``
            when hand labels and pseudo-labels are mixed).
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        channels: Sequence[str],
        norm: Norm,
        train: bool,
        crop: int = 256,
        label_column: str = "label",
    ) -> None:
        _check_channels(channels)
        self.rows = rows.reset_index(drop=True)
        self.channels = list(channels)
        self.norm = norm
        self.train = train
        self.crop = crop
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image = self.norm.apply(load_image(row, self.channels), self.channels)
        label = load_label(row[self.label_column])
        if self.train:
            top, left = random_crop_box(label, self.crop, None)
            image = image[:, top : top + self.crop, left : left + self.crop]
            label = label[top : top + self.crop, left : left + self.crop]
            image, label = augment(image, label, None)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "event": str(row["event"]),
            "chip_id": str(row["chip_id"]),
        }


def compute_norm(rows: pd.DataFrame, channels: Sequence[str], max_chips: int = 0) -> Norm:
    """Mean and standard deviation per channel over all finite pixels of the given chips."""
    sums = {c: 0.0 for c in channels}
    sq = {c: 0.0 for c in channels}
    counts = {c: 0 for c in channels}
    subset = rows if not max_chips else rows.head(max_chips)
    for row in subset.itertuples():
        image = load_image(row._asdict(), channels)
        for i, name in enumerate(channels):
            values = image[i][np.isfinite(image[i])].astype(np.float64)
            sums[name] += float(values.sum())
            sq[name] += float((values**2).sum())
            counts[name] += values.size
    mean = {c: sums[c] / counts[c] for c in channels}
    std = {c: float(np.sqrt(max(sq[c] / counts[c] - mean[c] ** 2, 1e-12))) for c in channels}
    return Norm(mean=mean, std=std)
