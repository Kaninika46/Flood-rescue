"""Per-pixel random forest baseline (E2).

Features per pixel (``FEATURES``): VV, VH, VV - VH (dB), HAND (m), slope (degrees), and the
5x5 mean and standard deviation of VV and VH. Sen1Floods11 has no pre-flood image, so there
are no change features.

Training samples a class-balanced set of pixels from ``s11_train`` only; hyper-parameters are
chosen on ``s11_val``; test splits are scored with :mod:`src.common.metrics` by
``scripts/evaluate.py``. scikit-learn's forest handles the rare NaN features natively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier

from src.level1_radar.datasets import load_image, load_label

FEATURES: tuple[str, ...] = (
    "VV",
    "VH",
    "VV_minus_VH",
    "HAND",
    "SLOPE",
    "VV_mean5",
    "VV_std5",
    "VH_mean5",
    "VH_std5",
)


def local_mean_std(band: np.ndarray, size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """``size x size`` mean and standard deviation ignoring NaN (NaN where no valid neighbour)."""
    valid = np.isfinite(band)
    filled = np.where(valid, band, 0.0).astype(np.float64)
    count = uniform_filter(valid.astype(np.float64), size, mode="nearest")
    s1 = uniform_filter(filled, size, mode="nearest")
    s2 = uniform_filter(filled**2, size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / count
        var = np.maximum(s2 / count - mean**2, 0.0)
    mean[count == 0] = np.nan
    var[count == 0] = np.nan
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def pixel_features(image: np.ndarray) -> np.ndarray:
    """``(H, W, F)`` feature cube from a ``(4, H, W)`` image ordered VV, VH, HAND, SLOPE."""
    vv, vh, hand, slope = image
    vv_mean, vv_std = local_mean_std(vv)
    vh_mean, vh_std = local_mean_std(vh)
    cube = np.stack([vv, vh, vv - vh, hand, slope, vv_mean, vv_std, vh_mean, vh_std], axis=-1)
    return cube.astype(np.float32)


def chip_features(row: Mapping[str, Any]) -> np.ndarray:
    """Feature cube of one split row."""
    return pixel_features(load_image(row, ["VV", "VH", "HAND", "SLOPE"]))


def sample_balanced(
    rows: pd.DataFrame, per_class_per_chip: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal numbers of water and non-water labelled pixels from every chip.

    Returns features ``(N, F)``, labels ``(N,)`` and the event of each sample.
    """
    xs, ys, events = [], [], []
    for row in rows.itertuples():
        row_dict = row._asdict()
        label = load_label(row.label)
        cube = chip_features(row_dict)
        for cls in (0, 1):
            idx = np.flatnonzero(label.ravel() == cls)
            if idx.size == 0:
                continue
            take = rng.choice(idx, size=min(per_class_per_chip, idx.size), replace=False)
            xs.append(cube.reshape(-1, cube.shape[-1])[take])
            ys.append(np.full(take.size, cls, np.uint8))
            events.append(np.full(take.size, row.event, dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(events)


def train_forest(
    x: np.ndarray, y: np.ndarray, params: Mapping[str, Any], seed: int
) -> RandomForestClassifier:
    """Fit a forest with ``params`` (n_estimators, max_depth, min_samples_leaf, ...)."""
    forest = RandomForestClassifier(random_state=seed, n_jobs=-1, **dict(params))
    forest.fit(x, y)
    return forest


def shift_to_half(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Monotone recalibration that moves ``threshold`` to 0.5 (a shift in log-odds).

    The forest is trained on class-balanced pixels, so its probabilities overstate water; the
    decision threshold tuned on validation pixels is folded in, and every method can then be
    scored at 0.5.
    """
    if threshold == 0.5:
        return prob
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)) - np.log(threshold / (1 - threshold))
    return (1 / (1 + np.exp(-logit))).astype(np.float32)


def predict_chip(
    forest: RandomForestClassifier, row: Mapping[str, Any], threshold: float = 0.5
) -> np.ndarray:
    """Water probability ``(H, W)`` for one chip, recalibrated so 0.5 is the tuned cut."""
    cube = chip_features(row)
    h, w, f = cube.shape
    prob = forest.predict_proba(cube.reshape(-1, f))[:, 1].reshape(h, w).astype(np.float32)
    return shift_to_half(prob, threshold)


def importances(forest: RandomForestClassifier, names: Sequence[str] = FEATURES) -> pd.DataFrame:
    """Impurity-based feature importances, largest first."""
    return (
        pd.DataFrame({"feature": list(names), "importance": forest.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- command line


def _grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    import itertools

    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _val_sample(
    rows: pd.DataFrame, per_chip: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random labelled pixels per chip (not balanced): pooled IoU on them estimates split IoU."""
    xs, ys, events = [], [], []
    for row in rows.itertuples():
        label = load_label(row.label).ravel()
        idx = np.flatnonzero(label >= 0)
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(per_chip, idx.size), replace=False)
        cube = chip_features(row._asdict())
        xs.append(cube.reshape(-1, cube.shape[-1])[take])
        ys.append(label[take])
        events.append(np.full(take.size, row.event, dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(events)


def main(
    config: str,
    seed: int = 0,
    overrides: Sequence[str] = (),
    smoke: bool = False,
) -> None:
    """Train one E2 forest: sample, grid-search on validation pixels, save model + importances."""
    import joblib

    from src.common.config import data_path, load_config, run_name, save_config
    from src.common.logging import start_run
    from src.common.metrics import SplitScorer
    from src.common.seed import seed_everything
    from src.level1_radar.s11 import read_split

    cfg = load_config(config, overrides)
    seed_everything(seed)
    rf_cfg = cfg["rf"]
    train_rows = read_split(cfg["data"]["train_split"])
    val_rows = read_split(cfg["data"]["val_split"])
    if smoke:
        train_rows, val_rows = train_rows.head(2), val_rows.head(2)
    out_dir = data_path(
        "checkpoints", run_name(cfg["experiment"], seed) + ("-smoke" if smoke else "")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with start_run(
        cfg, seed, job_type="train", smoke=smoke, save_config_to=out_dir / "config.yaml"
    ) as run:
        x, y, _ = sample_balanced(train_rows, rf_cfg["per_class_per_chip"], rng)
        xv, yv, ev = _val_sample(val_rows, rf_cfg["val_pixels_per_chip"], rng)
        run.log_summary({"train_pixels": len(y), "val_pixels": len(yv)})
        best, best_iou, best_params = None, -1.0, {}
        for i, params in enumerate(_grid(rf_cfg["grid"])):
            full = {**rf_cfg.get("base", {}), **params}
            if smoke:
                full["n_estimators"] = 5
            forest = train_forest(x, y, full, seed)
            prob = forest.predict_proba(xv)[:, 1]
            # training pixels are class-balanced, so the best cut is usually above 0.5
            for threshold in rf_cfg.get("thresholds", [0.5]):
                scorer = SplitScorer()
                for event in np.unique(ev):
                    sel = ev == event
                    pred = (prob[sel] >= threshold).astype(np.uint8)
                    scorer.update(pred, yv[sel], event=str(event))
                iou = scorer.overall.iou
                print(f"  {params} threshold {threshold} -> val IoU {iou:.4f}")
                if iou > best_iou:
                    best, best_iou = forest, iou
                    best_params = {**full, "decision_threshold": float(threshold)}
            run.log_metrics({"val/best_iou_so_far": best_iou}, step=i)
        joblib.dump(best, out_dir / "forest.joblib", compress=3)
        importances(best).to_csv(out_dir / "feature_importance.csv", index=False)
        save_config(best_params, out_dir / "best_params.yaml")
        run.log_summary({"val/best_iou": best_iou})
        run.log_file(out_dir / "feature_importance.csv")
        print(f"best {best_params} val IoU {best_iou:.4f} -> {out_dir}")


if __name__ == "__main__":
    import typer

    def _cli(
        config: str = typer.Option(..., help="configs/E2.yaml"),
        seed: int = typer.Option(0),
        set_: list[str] = typer.Option(
            [], "--set", help="Override, e.g. rf.per_class_per_chip=500"
        ),
        smoke: bool = typer.Option(False, help="2 chips, 5 trees: checks the pipeline end to end."),
    ) -> None:
        main(config, seed, set_, smoke)

    typer.run(_cli)
