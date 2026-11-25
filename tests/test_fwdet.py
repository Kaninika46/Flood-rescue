"""FwDET port on synthetic terrain where the true depth is known."""

from __future__ import annotations

import numpy as np

from src.depth.fwdet import (
    FwdetSettings,
    boundary_cells,
    clean_mask,
    focal_mean_masked,
    fwdet,
)


def _bowl(n: int = 121, pixel: float = 10.0, k: float = 2e-5) -> tuple[np.ndarray, float]:
    """Paraboloid DEM z = 100 + k * r^2 (metres), r from the centre."""
    yy, xx = (np.mgrid[0:n, 0:n] - n // 2) * pixel
    return 100.0 + k * (xx**2 + yy**2), pixel


def test_bowl_depth_matches_truth() -> None:
    dem, pixel = _bowl()
    level = 105.0
    flood = dem < level
    truth = np.where(flood, level - dem, np.nan)
    res = fwdet(flood, dem, pixel, pixel, FwdetSettings(lowpass_passes=0))
    assert np.isnan(res.depth[~flood]).all()
    err = np.abs(res.depth[flood] - truth[flood])
    # flood radius 500 m; boundary cells sit up to one cell (10 m, about 0.2 m of height)
    # inside the shoreline, so the estimate is slightly low but tightly bounded
    assert np.median(err) < 0.2
    assert err.max() < 0.4
    assert abs(float(np.nanmax(res.depth)) - 5.0) < 0.4


def test_tilted_valley_keeps_tilted_water_surface() -> None:
    # valley along x with a downstream slope; water 2 m deep everywhere it floods
    n, pixel = 80, 10.0
    yy, xx = np.mgrid[0:n, 0:n] * pixel
    floor = 50.0 - 0.001 * xx
    dem = floor + 0.01 * np.abs(yy - yy.mean())
    level = floor + 2.0
    flood = dem < level
    res = fwdet(flood, dem, pixel, pixel, FwdetSettings(smooth_iterations=2, lowpass_passes=0))
    inner = flood & (xx > 100) & (xx < 690)
    assert np.allclose(res.depth[inner], 2.0 - (dem - floor)[inner], atol=0.35)


def test_depth_never_negative_and_nodata_outside() -> None:
    dem, pixel = _bowl(61, k=2e-4)
    flood = dem < 102.0
    flood[30, 30] = False  # one-pixel hole gets filled
    res = fwdet(flood, dem, pixel, pixel)
    assert res.flood[30, 30]
    assert np.nanmin(res.depth) >= 0
    assert np.isnan(res.depth[0, 0])


def test_clean_mask_fills_small_holes_and_drops_specks() -> None:
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2:18] = True
    mask[8:10, 8:10] = False  # 4-pixel hole
    mask[0, 19] = True  # 1-pixel speck
    out = clean_mask(mask, min_hole_px=10, min_object_px=5)
    assert out[8:10, 8:10].all()
    assert not out[0, 19]


def test_boundary_excludes_raster_edge() -> None:
    flood = np.zeros((6, 6), bool)
    flood[:, :3] = True
    b = boundary_cells(flood)
    assert b[:, 2].all() and not b[:, 0].any()


def test_focal_mean_masked_ignores_unmasked() -> None:
    values = np.array([[1.0, 100.0, 3.0]])
    mask = np.array([[True, False, True]])
    out = focal_mean_masked(values, mask, 3)
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0


def test_ocean_boundary_removed() -> None:
    dem = np.tile(np.linspace(-2, 10, 50), (40, 1))
    flood = dem < 3.0
    res = fwdet(flood, dem, 10.0, 10.0, FwdetSettings(min_slope_pct=0))
    assert not res.boundary[:, dem[0] <= 0.5].any()


def test_steep_boundary_option() -> None:
    dem, pixel = _bowl(61, k=0.05)
    flood = dem < 120.0
    plain = fwdet(flood, dem, pixel, pixel, FwdetSettings())
    capped = fwdet(flood, dem, pixel, pixel, FwdetSettings(max_slope_deg=5.0))
    assert plain.boundary.sum() > 0 and capped.boundary.sum() == 0
    assert np.isnan(capped.depth).all()
