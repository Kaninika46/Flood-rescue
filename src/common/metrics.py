"""Flood-class segmentation metrics: IoU, F1, precision and recall.

Rules:

* Positive class is 1 (water / flood). Label -1 is no-data and is ignored.
* TP, FP and FN are **summed over the whole split** and the metric is computed once from the
  totals. Never average per-chip IoU: chips with little water would dominate.
* Every split is also reported per flood event.

Conventions used here:

* Predictions must already be binary (bool, or integers 0/1). Threshold probabilities with
  :func:`binarize` first; float predictions are refused so a forgotten threshold cannot
  pass silently.
* A prediction of -1 (method produced no output) counts as "not water". This keeps every
  method scored on the same pixels. To drop such pixels instead, pass ``valid_mask``.
* Any other value (for example a 0/255 mask) raises ``ValueError``.
* A metric whose denominator is zero is ``NaN``, never 0 or 1. For example IoU is NaN on a
  split where neither labels nor predictions contain water. Report NaN as "n/a".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

IGNORE_INDEX = -1
POSITIVE = 1
METRIC_NAMES: tuple[str, ...] = ("iou", "f1", "precision", "recall")


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


@dataclass(frozen=True)
class Counts:
    """Pixel counts for the positive (flood) class over valid pixels."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @property
    def n_valid(self) -> int:
        """Number of scored (non-ignored) pixels."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def iou(self) -> float:
        """Intersection over union of the flood class: TP / (TP + FP + FN)."""
        return _safe_div(self.tp, self.tp + self.fp + self.fn)

    @property
    def precision(self) -> float:
        """TP / (TP + FP)."""
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """TP / (TP + FN)."""
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        """2 TP / (2 TP + FP + FN), equal to the Dice coefficient."""
        return _safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def as_dict(self) -> dict[str, float | int]:
        """Metrics and raw counts as a flat dict (good for logging and tables)."""
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n_valid": self.n_valid,
        }


def _to_numpy(x: Any) -> np.ndarray:
    """Accept NumPy arrays, lists and PyTorch tensors (on any device)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def binarize(prob: Any, threshold: float = 0.5) -> np.ndarray:
    """Turn a probability map into a 0/1 ``uint8`` prediction (``prob >= threshold``).

    NaN probabilities become 0.
    """
    prob = _to_numpy(prob)
    with np.errstate(invalid="ignore"):
        return (prob >= threshold).astype(np.uint8)


def _check_values(name: str, values: np.ndarray, allowed: Sequence[int]) -> None:
    bad = ~np.isin(values, allowed)
    if bad.any():
        found = np.unique(values[bad])[:5].tolist()
        raise ValueError(
            f"{name} contains values {found}; allowed are {list(allowed)}. "
            "Masks saved as 0/255 must be converted to 0/1 first."
        )


def confusion_counts(
    pred: Any,
    label: Any,
    *,
    valid_mask: Any | None = None,
    ignore_index: int = IGNORE_INDEX,
    positive: int = POSITIVE,
    check_values: bool = True,
) -> Counts:
    """Count TP, FP, FN and TN for the flood class on one chip, tile or batch.

    Args:
        pred: Binary prediction (bool or integer 0/1; -1 counts as not water). Any shape.
        label: Reference labels, same shape: ``ignore_index`` (no data), 0 or ``positive``.
        valid_mask: Optional boolean mask, same shape; False pixels are not scored.
        ignore_index: Label value to ignore.
        positive: Value of the flood class in both ``pred`` and ``label``.
        check_values: Refuse unexpected values (e.g. 255). Leave on.

    Returns:
        The pixel counts over the valid pixels.
    """
    pred = _to_numpy(pred)
    label = _to_numpy(label)
    if pred.shape != label.shape:
        raise ValueError(f"pred shape {pred.shape} != label shape {label.shape}")
    if pred.dtype.kind in "fc":
        raise TypeError(
            f"pred has float dtype {pred.dtype}; threshold probabilities with binarize() first"
        )
    if check_values:
        _check_values("label", label, (ignore_index, 0, positive))
        if pred.dtype != bool:
            _check_values("pred", pred, (ignore_index, 0, positive))

    valid = label != ignore_index
    if valid_mask is not None:
        valid_mask = _to_numpy(valid_mask)
        if valid_mask.shape != label.shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} != label shape {label.shape}")
        valid &= valid_mask.astype(bool)

    pred_pos = (pred == positive) & valid
    label_pos = (label == positive) & valid
    tp = int(np.count_nonzero(pred_pos & label_pos))
    fp = int(np.count_nonzero(pred_pos & ~label_pos))
    fn = int(np.count_nonzero(label_pos & ~pred_pos))
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    return Counts(tp=tp, fp=fp, fn=fn, tn=tn)


class SplitScorer:
    """Accumulate counts over a whole split, overall and per flood event.

    Example::

        scorer = SplitScorer()
        for chip in split:
            scorer.update(binarize(prob), chip.label, event=chip.event)
        scorer.overall.iou        # pooled over the split
        scorer.per_event["Bolivia"].f1
        run.log_metrics(scorer.flat(prefix="test/"))
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, positive: int = POSITIVE) -> None:
        self.ignore_index = ignore_index
        self.positive = positive
        self.overall = Counts()
        self.per_event: dict[str, Counts] = {}
        self.n_items = 0

    def add(self, counts: Counts, event: str | None = None) -> None:
        """Add precomputed counts (e.g. from another process) to the totals."""
        self.overall = self.overall + counts
        self.n_items += 1
        if event is not None:
            self.per_event[event] = self.per_event.get(event, Counts()) + counts

    def update(
        self,
        pred: Any,
        label: Any,
        event: str | Sequence[str] | None = None,
        valid_mask: Any | None = None,
    ) -> Counts:
        """Score one chip, or a batch along axis 0 when ``event`` is a list (one per item).

        Returns:
            The counts added by this call.
        """
        kwargs = {"ignore_index": self.ignore_index, "positive": self.positive}
        if event is None or isinstance(event, str):
            counts = confusion_counts(pred, label, valid_mask=valid_mask, **kwargs)
            self.add(counts, event)
            return counts
        pred, label = _to_numpy(pred), _to_numpy(label)
        masks = None if valid_mask is None else _to_numpy(valid_mask)
        if len(event) != pred.shape[0]:
            raise ValueError(f"{len(event)} events for a batch of {pred.shape[0]} items")
        total = Counts()
        for i, item_event in enumerate(event):
            mask = None if masks is None else masks[i]
            counts = confusion_counts(pred[i], label[i], valid_mask=mask, **kwargs)
            self.add(counts, item_event)
            total = total + counts
        return total

    def summary(self) -> dict[str, Any]:
        """Nested dict: ``{"overall": {...}, "per_event": {event: {...}}}``."""
        return {
            "overall": self.overall.as_dict(),
            "per_event": {e: self.per_event[e].as_dict() for e in sorted(self.per_event)},
        }

    def flat(self, prefix: str = "") -> dict[str, float | int]:
        """Flat dict for experiment logging, e.g. ``test/iou`` and ``test/event/Bolivia/iou``."""
        out: dict[str, float | int] = {f"{prefix}{k}": v for k, v in self.overall.as_dict().items()}
        for event in sorted(self.per_event):
            for key, value in self.per_event[event].as_dict().items():
                out[f"{prefix}event/{event}/{key}"] = value
        return out


def score_split(
    preds: Iterable[Any], labels: Iterable[Any], events: Iterable[str] | None = None
) -> SplitScorer:
    """Score paired iterables of binary predictions and labels (optionally with events)."""
    scorer = SplitScorer()
    if events is None:
        for pred, label in zip(preds, labels, strict=True):
            scorer.update(pred, label)
    else:
        for pred, label, event in zip(preds, labels, events, strict=True):
            scorer.update(pred, label, event=event)
    return scorer
