"""Tests for src/common/metrics.py on tiny hand-made arrays."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from src.common.metrics import (
    Counts,
    SplitScorer,
    binarize,
    confusion_counts,
    score_split,
)


def test_known_counts_on_2x2() -> None:
    # label: water, water / land, land ; pred: water, land / water, land
    label = np.array([[1, 1], [0, 0]], dtype=np.int16)
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    c = confusion_counts(pred, label)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.iou == pytest.approx(1 / 3)
    assert c.precision == pytest.approx(0.5)
    assert c.recall == pytest.approx(0.5)
    assert c.f1 == pytest.approx(0.5)


def test_perfect_prediction() -> None:
    label = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int16)
    c = confusion_counts(label.copy(), label)
    assert c.iou == 1.0 and c.f1 == 1.0 and c.precision == 1.0 and c.recall == 1.0


def test_ignore_label_is_never_scored() -> None:
    label = np.array([[-1, -1], [1, 0]], dtype=np.int16)
    # predictions over the no-data pixels must not count, whatever they are
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    c = confusion_counts(pred, label)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 0, 0, 1)
    assert c.n_valid == 2


def test_everything_no_data_gives_nan_without_warnings() -> None:
    label = np.full((4, 4), -1, dtype=np.int16)
    pred = np.ones((4, 4), dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = confusion_counts(pred, label)
        values = c.as_dict()
    assert (c.tp, c.fp, c.fn, c.tn, c.n_valid) == (0, 0, 0, 0, 0)
    for name in ("iou", "f1", "precision", "recall"):
        assert math.isnan(values[name]), name


def test_everything_no_data_over_a_whole_split() -> None:
    scorer = SplitScorer()
    for event in ("Bolivia", "India"):
        scorer.update(np.zeros((3, 3), np.uint8), np.full((3, 3), -1, np.int16), event=event)
    assert scorer.overall.n_valid == 0
    assert math.isnan(scorer.overall.iou)
    assert all(math.isnan(c.iou) for c in scorer.per_event.values())


def test_no_water_anywhere_is_undefined_not_perfect() -> None:
    label = np.zeros((2, 2), dtype=np.int16)
    pred = np.zeros((2, 2), dtype=np.uint8)
    c = confusion_counts(pred, label)
    assert c.tn == 4
    assert math.isnan(c.iou) and math.isnan(c.recall) and math.isnan(c.precision)


def test_split_iou_sums_counts_instead_of_averaging_chips() -> None:
    # chip A: one water pixel, missed -> IoU 0
    label_a = np.array([[1, 0, 0, 0]], dtype=np.int16)
    pred_a = np.array([[0, 0, 0, 0]], dtype=np.uint8)
    # chip B: three water pixels, all found -> IoU 1
    label_b = np.array([[1, 1, 1, 0]], dtype=np.int16)
    pred_b = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    scorer = score_split([pred_a, pred_b], [label_a, label_b])
    # pooled: TP 3, FP 0, FN 1 -> 0.75 ; mean of per-chip IoU would be 0.5
    assert scorer.overall.iou == pytest.approx(0.75)
    assert scorer.n_items == 2


def test_per_event_breakdown_adds_up_to_overall() -> None:
    scorer = SplitScorer()
    scorer.update(np.array([[1, 0]], np.uint8), np.array([[1, 1]], np.int16), event="Bolivia")
    scorer.update(np.array([[1, 1]], np.uint8), np.array([[0, 1]], np.int16), event="India")
    scorer.update(np.array([[0, 0]], np.uint8), np.array([[1, -1]], np.int16), event="India")
    bolivia, india = scorer.per_event["Bolivia"], scorer.per_event["India"]
    assert (bolivia.tp, bolivia.fp, bolivia.fn) == (1, 0, 1)
    assert (india.tp, india.fp, india.fn) == (1, 1, 1)
    assert bolivia + india == scorer.overall
    summary = scorer.summary()
    assert list(summary["per_event"]) == ["Bolivia", "India"]
    flat = scorer.flat(prefix="test/")
    assert flat["test/iou"] == pytest.approx(scorer.overall.iou)
    assert flat["test/event/India/fn"] == 1


def test_batch_update_with_one_event_per_item() -> None:
    labels = np.array([[[1, 0]], [[1, 1]]], dtype=np.int16)  # (N=2, H=1, W=2)
    preds = np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8)
    scorer = SplitScorer()
    scorer.update(preds, labels, event=["Ghana", "Spain"])
    assert scorer.per_event["Ghana"].iou == 1.0
    assert scorer.per_event["Spain"].iou == pytest.approx(0.5)
    with pytest.raises(ValueError, match="events for a batch"):
        scorer.update(preds, labels, event=["Ghana"])


def test_bool_predictions_and_valid_mask() -> None:
    label = np.array([[1, 1, 0]], dtype=np.int16)
    pred = np.array([[True, False, True]])
    mask = np.array([[True, False, True]])
    c = confusion_counts(pred, label, valid_mask=mask)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 0, 0)


def test_prediction_minus_one_counts_as_not_water() -> None:
    label = np.array([[1, 0]], dtype=np.int16)
    pred = np.array([[-1, -1]], dtype=np.int16)
    c = confusion_counts(pred, label)
    assert (c.tp, c.fp, c.fn, c.tn) == (0, 0, 1, 1)


def test_float_predictions_are_refused_until_binarized() -> None:
    label = np.array([[1, 0]], dtype=np.int16)
    prob = np.array([[0.7, 0.2]], dtype=np.float32)
    with pytest.raises(TypeError, match="binarize"):
        confusion_counts(prob, label)
    assert confusion_counts(binarize(prob), label).iou == 1.0


def test_binarize_threshold_and_nan() -> None:
    prob = np.array([0.49, 0.5, 0.9, np.nan])
    assert binarize(prob).tolist() == [0, 1, 1, 0]
    assert binarize(prob, threshold=0.95).tolist() == [0, 0, 0, 0]


def test_0_255_masks_are_refused() -> None:
    label = np.array([[255, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="0/255"):
        confusion_counts(np.array([[1, 0]], np.uint8), label)
    with pytest.raises(ValueError, match="pred contains"):
        confusion_counts(np.array([[255, 0]], np.uint8), np.array([[1, 0]], np.int16))


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        confusion_counts(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.int16))


def test_counts_add_and_defaults() -> None:
    assert Counts() + Counts(1, 2, 3, 4) == Counts(1, 2, 3, 4)
    assert Counts(tp=2, fp=1, fn=1).f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_torch_tensors_are_accepted() -> None:
    torch = pytest.importorskip("torch")
    label = torch.tensor([[1, 0, -1]], dtype=torch.int16)
    pred = torch.tensor([[1, 1, 1]], dtype=torch.uint8)
    c = confusion_counts(pred, label)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 0, 0)
