"""Level 2 scores that do not need the APLS code (B3).

* pixel IoU / precision / recall from prediction GeoTIFFs and the ground-truth masks
  (building, road, flooded building, flooded road), pooled over a split with
  :mod:`src.common.metrics`
* **flooded-road precision and recall by length** from vectors: recall = share of the
  ground-truth flooded road length that lies within ``buffer_m`` of a predicted flooded road;
  precision = share of predicted flooded road length within ``buffer_m`` of a truly flooded
  road. Also a count-based recall (a truly flooded segment counts as found when at least half
  of its length is covered). Flooded-road recall is the key safety number.

Works with Shapely 1.8 (sn8 env) and 2.x (flood env).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _length(geoms: Iterable[BaseGeometry]) -> float:
    return float(sum(g.length for g in geoms))


def length_precision_recall(
    pred: Sequence[BaseGeometry],
    truth: Sequence[BaseGeometry],
    buffer_m: float = 5.0,
    min_cover: float = 0.5,
) -> dict[str, float]:
    """Length-weighted precision / recall of predicted vs true line sets (metric CRS).

    Returns ``precision``, ``recall``, ``f1``, ``segment_recall`` and the lengths used.
    Empty sides give NaN for the metric they would divide by zero.
    """
    nan = float("nan")
    pred = [g for g in pred if g is not None and not g.is_empty]
    truth = [g for g in truth if g is not None and not g.is_empty]
    pred_len, truth_len = _length(pred), _length(truth)
    truth_zone = unary_union([g.buffer(buffer_m) for g in truth]) if truth else None
    pred_zone = unary_union([g.buffer(buffer_m) for g in pred]) if pred else None
    covered_pred = _length(g.intersection(truth_zone) for g in pred) if truth_zone else 0.0
    covered_truth = [g.intersection(pred_zone).length if pred_zone else 0.0 for g in truth]
    precision = covered_pred / pred_len if pred_len else nan
    recall = sum(covered_truth) / truth_len if truth_len else nan
    found = sum(
        1
        for g, c in zip(truth, covered_truth, strict=True)
        if g.length and c / g.length >= min_cover
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else nan
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "segment_recall": found / len(truth) if truth else nan,
        "pred_length_m": pred_len,
        "truth_length_m": truth_len,
        "n_pred": len(pred),
        "n_truth": len(truth),
    }


def is_flooded(value: Any) -> bool:
    """SpaceNet 8 / submission flood flags: "yes", "True", True, 1 -> True."""
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "true", "1")
    return bool(value) if value is not None and value == value else False


def tile_bounds(path: Path) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a pre-event tile."""
    import rasterio

    with rasterio.open(path) as src:
        return tuple(src.bounds)


def lon_lat_order(geom: Any, bounds: tuple[float, float, float, float]) -> Any:
    """The baseline writes road ``WKT_Geo`` as (lat lon); swap when that puts it off its tile."""
    from shapely.geometry import box
    from shapely.ops import transform

    tile = box(*bounds).buffer(0.001)
    if tile.intersects(geom):
        return geom
    try:  # Shapely 2
        import shapely

        swapped = shapely.transform(geom, lambda coords: coords[:, ::-1])
    except AttributeError:  # Shapely 1.8 (sn8 env)
        swapped = transform(lambda x, y: (y, x), geom)
    return swapped if tile.intersects(swapped) else geom
