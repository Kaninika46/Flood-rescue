"""One interface for every Level 1 method: ``predictor(row) -> water probability (H, W)``.

``make_predictor(cfg, seed, variant)`` builds the predictor for an experiment config:

========================  ==================================================================
``method``                what the predictor does
========================  ==================================================================
``otsu``                  shipped Sen1Floods11 Otsu labels, or our Otsu on VH (``variants``)
``random_forest``         forest saved by ``python -m src.level1_radar.random_forest``
``unet``                  ensemble of U-Net checkpoints of this experiment (one per seed)
``etci``                  the released ETCI 2021 ensemble with the dB -> PNG input mapping
``fusion``                stacking model saved by ``python -m src.level1_radar.fusion``
========================  ==================================================================

Binary methods return 0.0/1.0 so every method can be thresholded the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from src.common.config import data_path, run_name
from src.level1_radar.datasets import load_image, load_label
from src.level1_radar.otsu import OtsuSettings, otsu_water

Predictor = Callable[[Mapping[str, Any]], np.ndarray]


def checkpoint_dir(experiment: str, seed: int) -> Any:
    """``$DATA_ROOT/checkpoints/<experiment>-s<seed>``."""
    return data_path("checkpoints", run_name(experiment, seed))


def otsu_predictor(settings: Mapping[str, Any]) -> Predictor:
    """Shipped Otsu labels (``source: shipped``) or Otsu computed on one band."""
    if settings.get("source") == "shipped":

        def shipped(row: Mapping[str, Any]) -> np.ndarray:
            return (load_label(row["otsu"]) == 1).astype(np.float32)

        return shipped

    opts = OtsuSettings(
        band=settings.get("band", "VH"),
        max_threshold_db=settings.get("max_threshold_db"),
        max_hand_m=settings.get("max_hand_m"),
        max_slope_deg=settings.get("max_slope_deg"),
    )
    terrain = opts.max_hand_m is not None or opts.max_slope_deg is not None
    per_event = settings.get("threshold_scope", "chip") == "event"

    def computed(row: Mapping[str, Any]) -> np.ndarray:
        channels = [opts.band, "HAND", "SLOPE"] if terrain else [opts.band]
        image = load_image(row, channels)
        hand = image[1] if terrain else None
        slope = image[2] if terrain else None
        threshold = event_threshold(str(row["event"]), opts.band) if per_event else None
        water, _ = otsu_water(image[0], opts, hand=hand, slope=slope, threshold=threshold)
        return water.astype(np.float32)

    return computed


@lru_cache(maxsize=64)
def event_threshold(event: str, band: str, pixels_per_chip: int = 20000) -> float:
    """One Otsu threshold for a whole flood event, like a scene-level threshold.

    Pools a fixed random sample of pixel values from every hand-labelled chip of the event
    (images only; labels are never read), so chips without water get a sensible threshold.
    """
    import pandas as pd

    from src.level1_radar.otsu import otsu_threshold
    from src.level1_radar.s11 import read_split

    rows = pd.concat([read_split(n) for n in ("s11_train", "s11_val", "s11_test", "s11_bolivia")])
    rng = np.random.default_rng(0)
    values = []
    for row in rows[rows["event"] == event].to_dict("records"):
        band_values = load_image(row, [band])[0].ravel()
        band_values = band_values[np.isfinite(band_values)]
        if band_values.size:
            take = min(pixels_per_chip, band_values.size)
            values.append(rng.choice(band_values, size=take, replace=False))
    return otsu_threshold(np.concatenate(values)) if values else float("nan")


def forest_predictor(experiment: str, seed: int) -> Predictor:
    """Random forest of one seed."""
    import joblib

    from src.common.config import load_yaml
    from src.level1_radar.random_forest import predict_chip

    folder = checkpoint_dir(experiment, seed)
    forest = joblib.load(folder / "forest.joblib")
    threshold = float(load_yaml(folder / "best_params.yaml").get("decision_threshold", 0.5))
    return lambda row: predict_chip(forest, row, threshold)


@lru_cache(maxsize=8)
def _unet_ensemble(paths: tuple[str, ...], tta: bool) -> Any:
    from src.level1_radar.predict import Ensemble

    return Ensemble.from_checkpoints(list(paths), tta=tta)


def unet_predictor(paths: Sequence[str], tta: bool = False, return_std: bool = False) -> Predictor:
    """U-Net ensemble over the given checkpoints (mean probability, or the std if asked)."""
    ensemble = _unet_ensemble(tuple(str(p) for p in paths), tta)

    def predict(row: Mapping[str, Any]) -> np.ndarray:
        layers = ensemble.load_layers(row)
        mean, std = ensemble.predict(layers)
        return std if return_std else mean

    return predict


def etci_predictor(cfg: Mapping[str, Any]) -> Predictor:
    """Released ETCI ensemble: on ETCI PNG tiles (E0) or on Sen1Floods11 dB chips (E3)."""
    from src.level1_radar.etci_reproduce import EtciEnsemble, etci_row_arrays, load_scaling

    ensemble = EtciEnsemble.from_config(cfg["etci"])
    scaling = load_scaling(cfg["etci"]["scaling"]) if cfg["etci"].get("scaling") else None

    def predict(row: Mapping[str, Any]) -> np.ndarray:
        if "vv_png" in row or "vv_cell" in row:
            vv01, vh01, _ = etci_row_arrays(row)
            return ensemble.predict_png(vv01, vh01)
        if scaling is None:
            raise ValueError("etci.scaling is needed to run ETCI models on dB chips (E3)")
        vv, vh = load_image(row, ["VV", "VH"])
        return ensemble.predict_db(vv, vh, scaling)

    return predict


def make_predictor(cfg: Mapping[str, Any], seed: int, variant: str | None = None) -> Predictor:
    """Predictor for ``cfg["method"]`` (and ``variant`` for multi-variant experiments)."""
    method = cfg["method"]
    if method == "otsu":
        return otsu_predictor(cfg["variants"][variant])
    if method == "random_forest":
        return forest_predictor(cfg["experiment"], seed)
    if method == "unet":
        from src.level1_radar.predict import member_paths

        if cfg.get("eval", {}).get("ensemble", False):
            paths = member_paths(cfg)  # ensemble.members, or every seed of the experiment
        else:
            paths = [checkpoint_dir(cfg["experiment"], seed) / "best.pt"]
        return unet_predictor(paths, tta=cfg.get("eval", {}).get("tta", False))
    if method == "etci":
        return etci_predictor(cfg)
    if method == "fusion":
        from src.level1_radar.fusion import fusion_predictor

        return fusion_predictor(cfg, seed)
    raise ValueError(f"unknown method {method!r}")
