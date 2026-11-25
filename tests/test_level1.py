"""Level 1 radar code on tiny synthetic chips (no real data, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from affine import Affine  # noqa: E402

from src.common.raster import write_raster  # noqa: E402
from src.level1_radar.datasets import Norm, Sen1Floods11Dataset, compute_norm  # noqa: E402
from src.level1_radar.etci_reproduce import db_to_unit, is_blank, s1_to_rgb  # noqa: E402
from src.level1_radar.losses import masked_dice_focal_loss  # noqa: E402
from src.level1_radar.models import (  # noqa: E402
    adapt_first_conv_weight,
    binary_head_from_softmax,
    make_network,
    transfer_state,
)
from src.level1_radar.otsu import change_otsu, otsu_water  # noqa: E402
from src.level1_radar.predict import Ensemble, Member, tile_starts, window_weights  # noqa: E402
from src.level1_radar.pseudo_label import pseudo_from_probs  # noqa: E402
from src.level1_radar.random_forest import FEATURES, local_mean_std, pixel_features  # noqa: E402

TRANSFORM = Affine(8.98e-05, 0, 86.0, 0, -8.98e-05, 26.5)


# --------------------------------------------------------------------------- loss


def test_loss_ignores_minus_one_pixels() -> None:
    target = torch.tensor([[[1, 0, -1, -1]]])
    logits = torch.tensor([[[[3.0, -3.0, 0.0, 0.0]]]])
    base = masked_dice_focal_loss(logits, target)
    changed = logits.clone()
    changed[..., 2:] = torch.tensor([50.0, -50.0])  # predictions on ignored pixels change
    assert torch.allclose(base, masked_dice_focal_loss(changed, target))


def test_loss_all_ignored_is_zero_with_gradient() -> None:
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = masked_dice_focal_loss(logits, torch.full((1, 2, 2), -1))
    loss.backward()
    assert loss.item() == 0.0 and logits.grad is not None


def test_loss_lower_for_better_prediction() -> None:
    target = torch.tensor([[[1, 1, 0, 0]]])
    good = masked_dice_focal_loss(torch.tensor([[[4.0, 4.0, -4.0, -4.0]]]), target)
    bad = masked_dice_focal_loss(torch.tensor([[[-4.0, -4.0, 4.0, 4.0]]]), target)
    assert good < bad


# --------------------------------------------------------------------------- dataset


@pytest.fixture()
def fake_chips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Two synthetic 64x64 chips (S1 in dB, label, HAND) under a temporary DATA_ROOT."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(0)
    rows = []
    for i, event in enumerate(["Ghana", "India"]):
        label = (rng.random((64, 64)) > 0.7).astype(np.int16)
        label[:4] = -1
        s1 = np.where(label == 1, -22.0, -9.0)[None].repeat(2, 0).astype(np.float32)
        s1[1] -= 6
        s1[0, 0, 0] = np.nan
        hand = np.where(label == 1, 0.5, 8.0).astype(np.float32)
        paths = {k: f"c{i}_{k}.tif" for k in ("s1", "label", "hand")}
        write_raster(
            tmp_path / paths["s1"], s1, crs="EPSG:4326", transform=TRANSFORM, nodata=float("nan")
        )
        write_raster(tmp_path / paths["label"], label, crs="EPSG:4326", transform=TRANSFORM)
        write_raster(tmp_path / paths["hand"], hand, crs="EPSG:4326", transform=TRANSFORM)
        rows.append({"chip_id": f"c{i}", "event": event, **paths})
    return pd.DataFrame(rows)


def test_dataset_train_crops_and_ignore(fake_chips: pd.DataFrame) -> None:
    norm = compute_norm(fake_chips, ["VV", "VH", "HAND"])
    ds = Sen1Floods11Dataset(fake_chips, ["VV", "VH", "HAND"], norm, train=True, crop=32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert item["label"].shape == (32, 32)
    assert set(item["label"].unique().tolist()) <= {-1, 0, 1}
    assert torch.isfinite(item["image"]).all()  # NaN became 0


def test_dataset_eval_full_chip_keeps_minus_one(fake_chips: pd.DataFrame) -> None:
    norm = Norm(mean={"VV": -10.0, "VH": -17.0}, std={"VV": 4.0, "VH": 5.0})
    item = Sen1Floods11Dataset(fake_chips, ["VV", "VH"], norm, train=False)[1]
    assert item["image"].shape == (2, 64, 64)
    assert (item["label"][:4] == -1).all()
    assert item["event"] == "India"


def test_norm_statistics(fake_chips: pd.DataFrame) -> None:
    norm = compute_norm(fake_chips, ["HAND"])
    assert 0.5 < norm.mean["HAND"] < 8.0 and norm.std["HAND"] > 0


# --------------------------------------------------------------------------- models


def test_first_conv_adaptation_keeps_and_averages() -> None:
    w = torch.arange(2 * 3 * 1 * 1, dtype=torch.float32).reshape(2, 3, 1, 1)
    new = adapt_first_conv_weight(w, ["VV", "VH", "RATIO"], ["VV", "VH", "HAND"])
    assert torch.allclose(new[:, 0], w[:, 0]) and torch.allclose(new[:, 1], w[:, 1])
    assert torch.allclose(new[:, 2], w.mean(dim=1))
    two = adapt_first_conv_weight(w, ["VV", "VH", "RATIO"], ["VV", "VH"])
    assert torch.allclose(two[:, 0], w[:, 0] * 1.5)


def test_softmax_head_becomes_equivalent_logit() -> None:
    weight, bias = torch.randn(2, 4, 1, 1), torch.randn(2)
    new_w, new_b = binary_head_from_softmax(weight, bias)
    feats = torch.randn(1, 4, 3, 3)
    two = torch.nn.functional.conv2d(feats, weight, bias).softmax(1)[:, 1]
    one = torch.sigmoid(torch.nn.functional.conv2d(feats, new_w, new_b))[:, 0]
    assert torch.allclose(two, one, atol=1e-6)


def test_transfer_from_two_class_three_channel_network() -> None:
    src = make_network("unet", "mobilenet_v2", in_channels=3, classes=2)
    dst = make_network("unet", "mobilenet_v2", in_channels=2, classes=1)
    transfer_state(dst, src.state_dict(), ["VV", "VH", "RATIO"], ["VV", "VH"])
    src.eval(), dst.eval()
    x = torch.randn(1, 2, 64, 64)
    x3 = torch.cat([x, torch.zeros(1, 1, 64, 64)], 1)  # 3rd channel 0 -> only VV, VH matter
    src_w = src.encoder.features[0][0].weight.data
    src.encoder.features[0][0].weight.data = torch.cat([src_w[:, :2] * 1.5, src_w[:, 2:]], 1)
    with torch.no_grad():
        p_src = src(x3).softmax(1)[:, 1]
        p_dst = torch.sigmoid(dst(x))[:, 0]
    assert torch.allclose(p_src, p_dst, atol=1e-4)


# --------------------------------------------------------------------------- prediction


def test_tiling_covers_everything() -> None:
    assert tile_starts(512, 512, 448) == [0]
    assert tile_starts(1000, 512, 448) == [0, 448, 488]
    w = window_weights(8, 2)
    assert w.max() == 1.0 and w.min() > 0 and w[4, 4] == 1.0


class _Identity(torch.nn.Module):
    """'Model' whose logit is the first input channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] * 10


def test_ensemble_on_large_image_matches_pixelwise_rule() -> None:
    norm = Norm(mean={"VV": 0.0}, std={"VV": 1.0})
    members = [Member(_Identity(), ["VV"], norm, "a"), Member(_Identity(), ["VV"], norm, "b")]
    ens = Ensemble(members, device=torch.device("cpu"), tile=64, overlap=16, tta=True)
    vv = np.where(np.arange(150)[None, :] < 75, 1.0, -1.0).repeat(130, 0).astype(np.float32)
    mean, std = ens.predict({"VV": vv})
    assert mean.shape == (130, 150)
    expected = 1 / (1 + np.exp(-10 * vv))
    np.testing.assert_allclose(mean, expected, atol=1e-4)
    assert np.allclose(std, 0)


# --------------------------------------------------------------------------- baselines


def test_otsu_bimodal_and_terrain() -> None:
    rng = np.random.default_rng(1)
    db = np.where(np.arange(100)[None] < 30, -24.0, -12.0) + rng.normal(0, 1, (100, 100))
    water, t = otsu_water(db.astype(np.float32))
    assert -22 < t < -14
    assert water[:, :30].mean() > 0.95 and water[:, 30:].mean() < 0.05
    hand = np.zeros_like(db)
    hand[:, :10] = 40.0
    from src.level1_radar.otsu import OtsuSettings

    masked, _ = otsu_water(db, OtsuSettings(max_hand_m=15.0), hand=hand)
    assert masked[:, :10].sum() == 0


def test_otsu_nan_and_cap() -> None:
    from src.level1_radar.otsu import OtsuSettings

    db = np.full((10, 10), -12.0, np.float32)
    db[0, 0] = np.nan
    water, t = otsu_water(db, OtsuSettings(max_threshold_db=-18.0))
    assert t == -18.0 and water.sum() == 0


def test_change_otsu_finds_new_water() -> None:
    before = np.full((50, 50), -10.0)
    after = before.copy()
    after[:, :20] = -22.0
    after += np.random.default_rng(0).normal(0, 0.5, after.shape)
    water, _ = change_otsu(before, after)
    assert water[:, :20].mean() > 0.95 and water[:, 25:].mean() < 0.05


def test_random_forest_features() -> None:
    image = np.stack(
        [np.full((7, 7), -10.0), np.full((7, 7), -17.0), np.ones((7, 7)), np.zeros((7, 7))]
    ).astype(np.float32)
    image[0, 3, 3] = np.nan
    cube = pixel_features(image)
    assert cube.shape == (7, 7, len(FEATURES))
    assert cube[0, 0, FEATURES.index("VV_minus_VH")] == pytest.approx(7.0)
    mean, std = local_mean_std(image[0])
    assert mean[3, 3] == pytest.approx(-10.0) and std[3, 3] == pytest.approx(0.0)


def test_pseudo_labels_need_confidence_and_agreement() -> None:
    probs = np.array([[[0.95, 0.95, 0.5, 0.02]], [[0.99, 0.6, 0.5, 0.05]]], dtype=np.float32)
    out = pseudo_from_probs(probs, high=0.9, low=0.1, require_agreement=True)
    assert out.tolist() == [[1, -1, -1, 0]]


def test_etci_encoding_and_scaling() -> None:
    vv = np.array([[0.5, 0.0]], np.float32)
    vh = np.array([[0.25, 0.0]], np.float32)
    rgb = s1_to_rgb(vv, vh)
    assert rgb.shape == (3, 1, 2)
    assert rgb[2, 0, 0] == pytest.approx(0.5) and rgb[2, 0, 1] == pytest.approx(1.0)  # 0/0 -> 0
    unit = db_to_unit(np.array([-40.0, -15.0, 10.0, np.nan]), -30.0, 0.0)
    assert unit.tolist() == [0.0, 0.5, 1.0, 0.0]
    assert is_blank(np.array([[0.0, 1.0]])) and not is_blank(np.array([[0.2, 1.0]]))
