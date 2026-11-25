"""Otsu threshold baselines (E1).

* :func:`otsu_water` thresholds VH (dB) with Otsu's method: pixels darker than the threshold are
  water. Optionally pixels with HAND above ``max_hand_m`` or slope above ``max_slope_deg`` are
  forced to "not water" (water cannot sit high above the drainage or on steep ground).
* The threshold is computed per image (chip or scene). On images with no water the histogram is
  unimodal and Otsu splits land in two; ``max_threshold_db`` caps the threshold to avoid that.
* :func:`change_otsu` is the change-detection variant used for Nepal, where a pre-flood image
  exists: Otsu on (after - before) VH; strong darkening is new water.

NaN pixels (no radar data) are returned as 0 (not water).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class OtsuSettings:
    """Settings for :func:`otsu_water` (all optional except the band)."""

    band: str = "VH"
    max_threshold_db: float | None = None
    max_hand_m: float | None = None
    max_slope_deg: float | None = None
    bins: int = 256


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold of the finite values; NaN when there are fewer than two distinct values."""
    finite = values[np.isfinite(values)]
    if finite.size < 2 or float(finite.min()) == float(finite.max()):
        return float("nan")
    return float(threshold_otsu(finite, nbins=bins))


def apply_terrain_mask(
    water: np.ndarray,
    hand: np.ndarray | None,
    slope: np.ndarray | None,
    max_hand_m: float | None,
    max_slope_deg: float | None,
) -> np.ndarray:
    """Set water to 0 where HAND or slope exceed their limits (NaN terrain is left alone)."""
    out = water.copy()
    with np.errstate(invalid="ignore"):
        if max_hand_m is not None and hand is not None:
            out[hand > max_hand_m] = 0
        if max_slope_deg is not None and slope is not None:
            out[slope > max_slope_deg] = 0
    return out


def otsu_water(
    db: np.ndarray,
    settings: OtsuSettings = OtsuSettings(),
    hand: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Water mask (uint8 0/1) and the threshold used, from one band in dB.

    Args:
        db: ``(H, W)`` backscatter in dB (VH by default), NaN for no data.
        settings: Threshold cap and terrain exclusion limits.
        hand, slope: Optional ``(H, W)`` terrain rasters on the same grid.
        threshold: Use this threshold instead of computing one (e.g. a split-wide value).
    """
    t = otsu_threshold(db, settings.bins) if threshold is None else threshold
    if settings.max_threshold_db is not None and (np.isnan(t) or t > settings.max_threshold_db):
        t = settings.max_threshold_db
    if np.isnan(t):
        return np.zeros(db.shape, np.uint8), t
    with np.errstate(invalid="ignore"):
        water = (db < t).astype(np.uint8)
    water = apply_terrain_mask(water, hand, slope, settings.max_hand_m, settings.max_slope_deg)
    return water, t


def change_otsu(
    before_db: np.ndarray,
    after_db: np.ndarray,
    max_hand_m: float | None = None,
    hand: np.ndarray | None = None,
    min_drop_db: float = 3.0,
) -> tuple[np.ndarray, float]:
    """New water from a pre/post pair: Otsu on ``after - before``; darkening below it is water.

    Requires at least ``min_drop_db`` of darkening, so a noisy unimodal difference does not turn
    half the scene into water. Returns the mask (uint8) and the threshold (dB).
    """
    diff = after_db - before_db
    t = otsu_threshold(diff)
    if np.isnan(t):
        return np.zeros(diff.shape, np.uint8), t
    t = min(t, -abs(min_drop_db))
    with np.errstate(invalid="ignore"):
        water = (diff < t).astype(np.uint8)
    return apply_terrain_mask(water, hand, None, max_hand_m, None), t
