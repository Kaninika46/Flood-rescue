"""PyTorch dataset of SpaceNet 8 tiles (see ``src/level2_roads/data.py`` for the file layout).

Adapted from ``third_party/spacenet8/baseline/datasets/datasets.py`` (Apache-2.0, commit 30ca30c).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.config import data_path
from src.level2_roads.data import flood_class_map, read_image, warp_post_to_pre


class SN8Dataset(Dataset):
    """Tiles of a split for the ``foundation`` (pre -> building, road speed) or ``flood``
    (pre + post -> 5 classes) task. ``crop`` gives random training windows; ``None`` gives
    full tiles (evaluation)."""

    def __init__(
        self, rows: pd.DataFrame, task: str, crop: int | None = None, with_targets: bool = True
    ) -> None:
        if task not in ("foundation", "flood"):
            raise ValueError(f"task must be foundation or flood, got {task!r}")
        self.rows = rows.reset_index(drop=True)
        self.task = task
        self.crop = crop
        self.with_targets = with_targets

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        if not self.crop or (height <= self.crop and width <= self.crop):
            return 0, 0, height, width
        top = int(torch.randint(0, height - self.crop + 1, (1,)))
        left = int(torch.randint(0, width - self.crop + 1, (1,)))
        return top, left, self.crop, self.crop

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: Mapping[str, Any] = self.rows.iloc[index]
        pre = read_image(data_path(row["preimg"]))
        _, height, width = pre.shape
        top, left, h, w = self._crop_box(height, width)
        sl = (slice(None), slice(top, top + h), slice(left, left + w))
        item: dict[str, Any] = {
            "preimg": torch.from_numpy(pre[sl].copy()),
            "tile_id": row["tile_id"],
            "aoi": row["aoi"],
            "preimg_path": row["preimg"],
        }
        if self.task == "flood":
            post = warp_post_to_pre(data_path(row["postimg"]), height, width)
            item["postimg"] = torch.from_numpy(post[sl].copy())
        if not self.with_targets:
            return item
        if self.task == "foundation":
            item["building"] = torch.from_numpy(read_image(data_path(row["building"]))[sl].copy())
            item["roadspeed"] = torch.from_numpy(read_image(data_path(row["roadspeed"]))[sl].copy())
        else:
            flood = read_image(data_path(row["flood"]))[sl]
            item["flood"] = torch.from_numpy(flood_class_map(flood))
        return item
