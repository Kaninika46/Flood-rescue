"""E7: stacking fusion of Otsu, random forest and U-Net (prompt A8).

Per-pixel features (``FEATURES``): the Otsu water mask (E1 variant), the random-forest water
probability (E2, same seed), the U-Net ensemble mean and standard deviation (E6, all seeds),
and HAND. Two stackers are compared, a logistic regression (on standardised features) and
LightGBM; they are trained on a random subsample of labelled ``s11_val`` pixels, and the better
one is chosen by **leave-one-event-out** cross-validation on ``s11_val`` (pooled IoU), so no
random chip split is made. The winner is refit on all sampled validation pixels.

    python -m src.level1_radar.fusion --config configs/E7.yaml --seed 0
    python scripts/evaluate.py --config configs/E7.yaml
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import load_config, repo_root
from src.common.logging import get_logger, start_run
from src.common.metrics import SplitScorer
from src.common.seed import seed_everything
from src.level1_radar.datasets import load_image, load_label

log = get_logger("fusion")
FEATURES = ("otsu", "rf_prob", "unet_mean", "unet_std", "hand")
app = typer.Typer(add_completion=False, help=__doc__)


def make_stacker(kind: str, cfg: Mapping[str, Any], seed: int) -> Any:
    """``logistic`` or ``lightgbm`` classifier."""
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbose=-1, **dict(cfg.get("lightgbm", {})))
    raise ValueError(f"unknown fusion model {kind!r}")


@lru_cache(maxsize=4)
def _component_cfg(path: str) -> dict[str, Any]:
    return load_config(repo_root() / path)


class FeatureBuilder:
    """Computes the fusion feature cube ``(H, W, F)`` of a chip from the component models."""

    def __init__(self, fusion_cfg: Mapping[str, Any], seed: int) -> None:
        from src.level1_radar.methods import (
            checkpoint_dir,
            forest_predictor,
            otsu_predictor,
            unet_predictor,
        )

        comps = fusion_cfg["components"]
        e1 = _component_cfg(comps["otsu"]["config"])
        self.otsu = otsu_predictor(e1["variants"][comps["otsu"]["variant"]])
        e2 = _component_cfg(comps["rf"]["config"])
        self.rf = forest_predictor(e2["experiment"], seed)
        e6 = _component_cfg(comps["unet"]["config"])
        paths = [checkpoint_dir(e6["experiment"], s) / "best.pt" for s in e6["seeds"]]
        self.unet_mean = unet_predictor(paths)
        self.unet_std = unet_predictor(paths, return_std=True)

    def __call__(self, row: Mapping[str, Any]) -> np.ndarray:
        hand = load_image(row, ["HAND"])[0]
        layers = [self.otsu(row), self.rf(row), self.unet_mean(row), self.unet_std(row), hand]
        return np.stack(layers, axis=-1).astype(np.float32)


def sample_pixels(
    builder: FeatureBuilder, rows: list[dict[str, Any]], per_chip: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random labelled pixels from each chip: features, labels and event of each pixel."""
    xs, ys, events = [], [], []
    for row in rows:
        label = load_label(row["label"]).ravel()
        idx = np.flatnonzero(label >= 0)
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(per_chip, idx.size), replace=False)
        cube = builder(row)
        xs.append(cube.reshape(-1, cube.shape[-1])[take])
        ys.append(label[take])
        events.append(np.full(take.size, row["event"], dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(events)


def leave_one_event_out_iou(
    kind: str, cfg: Mapping[str, Any], x: np.ndarray, y: np.ndarray, events: np.ndarray, seed: int
) -> float:
    """Pooled IoU over held-out events (each event predicted by a model trained on the others)."""
    scorer = SplitScorer()
    for event in np.unique(events):
        test = events == event
        model = make_stacker(kind, cfg, seed).fit(x[~test], y[~test])
        pred = (model.predict_proba(x[test])[:, 1] >= 0.5).astype(np.uint8)
        scorer.update(pred, y[test], event=str(event))
    return scorer.overall.iou


def fusion_predictor(cfg: Mapping[str, Any], seed: int) -> Any:
    """Predictor (row -> probability) of a trained E7 model."""
    from src.level1_radar.methods import checkpoint_dir

    saved = joblib.load(checkpoint_dir(cfg["experiment"], seed) / "fusion.joblib")
    builder = FeatureBuilder(cfg["fusion"], seed)
    model = saved["model"]

    def predict(row: Mapping[str, Any]) -> np.ndarray:
        cube = builder(row)
        h, w, f = cube.shape
        return model.predict_proba(cube.reshape(-1, f))[:, 1].reshape(h, w).astype(np.float32)

    return predict


@app.command()
def main(
    config: Path = typer.Option(..., help="configs/E7.yaml"),
    seed: int = typer.Option(0),
    smoke: bool = typer.Option(False, help="3 chips, quick models."),
) -> None:
    """Train both stackers, pick the better by leave-one-event-out IoU, save it."""
    from src.level1_radar.methods import checkpoint_dir
    from src.level1_radar.s11 import read_split

    cfg = load_config(config)
    seed_everything(seed)
    fcfg = cfg["fusion"]
    rows = read_split(fcfg["train_split"]).to_dict("records")
    if smoke:
        rows = rows[:3]
    rng = np.random.default_rng(seed)
    out = checkpoint_dir(cfg["experiment"], seed)
    out.mkdir(parents=True, exist_ok=True)
    with start_run(cfg, seed, job_type="train", smoke=smoke) as run:
        builder = FeatureBuilder(fcfg, seed)
        x, y, events = sample_pixels(builder, rows, fcfg["val_pixels_per_chip"], rng)
        scores = {}
        for kind in fcfg["models"]:
            scores[kind] = leave_one_event_out_iou(kind, fcfg, x, y, events, seed)
            run.log_summary({f"cv/{kind}/iou": scores[kind]})
            log.info("%s: leave-one-event-out IoU %.4f", kind, scores[kind])
        best = max(scores, key=lambda k: scores[k])
        model = make_stacker(best, fcfg, seed).fit(x, y)
        joblib.dump(
            {"model": model, "kind": best, "features": FEATURES, "cv_iou": scores},
            out / "fusion.joblib",
        )
        run.log_summary({"chosen_is_lightgbm": float(best == "lightgbm")})
    log.info("chose %s (%s); saved %s", best, scores, out / "fusion.joblib")


if __name__ == "__main__":
    app()
