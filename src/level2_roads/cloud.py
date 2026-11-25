"""Simple cloud / haze mask for post-event optical tiles (S3, prompt B6).

Rule (thresholds in ``configs/S3.yaml`` ``cloud``): a pixel is cloud or thick haze when it is
bright (mean of R, G, B above ``brightness_min``) and smooth (standard deviation of brightness in
a ``window`` x ``window`` neighbourhood below ``texture_max``). Clouds are bright and featureless;
bright roofs and roads are bright but edged, so the texture test keeps them out. The mask is
cleaned with a morphological opening and closing and patches smaller than ``min_area_px`` are
dropped. Road segments mostly under the mask get status "unknown", never "dry".

Check the thresholds by eye with ``python scripts/cloud_check.py`` (10 PNGs) before trusting it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

DEFAULTS = {"brightness_min": 170.0, "texture_max": 6.0, "window": 15, "min_area_px": 400}


def cloud_mask(rgb: np.ndarray, cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    """Boolean ``(H, W)`` cloud/haze mask from a ``(3, H, W)`` uint8-scaled image."""
    c = {**DEFAULTS, **(cfg or {})}
    bright = rgb[:3].astype(np.float32).mean(axis=0)
    size = int(c["window"])
    mean = ndimage.uniform_filter(bright, size)
    sq = ndimage.uniform_filter(bright**2, size)
    texture = np.sqrt(np.maximum(sq - mean**2, 0.0))
    mask = (bright > c["brightness_min"]) & (texture < c["texture_max"])
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        mask = np.isin(labels, np.flatnonzero(sizes >= c["min_area_px"]) + 1)
    return mask
