"""E0 and E3: the released ETCI 2021 ensemble (Paul and Ganju 2021) in our environment.

Ported from ``third_party/etci2021`` (Apache-2.0, commit 90d47ca): ``src/etci_dataset.py``
(``s1_to_rgb``) and ``notebooks/Ensemble_Inference.ipynb``. Changes are listed in
``docs/E0_notes.md``; the main ones: no ``ttach`` (the 8 D4 test-time transforms are done here),
no DataParallel, no CRF (``pydensecrf`` does not build for Python 3.11), probabilities instead of
an argmax so our metrics and fusion can use them.

Input encoding (unchanged): PNG values / 255 for VV and VH, and a third channel
``1 - clip(nan_to_num(VH / VV), 0, 1)``. Models: U-Net and U-Net++ with a MobileNetV2 encoder,
2 output classes. Logits are averaged over test-time transforms and members, then soft-maxed;
``argmax`` of the original equals ``p(flood) > 0.5`` here.

For Sen1Floods11 (E3) the dB values must first be mapped to the PNG scale; ETCI never published
that mapping, so ``fit-scaling`` estimates it (see ``docs/E3_notes.md``).

    python -m src.level1_radar.etci_reproduce download-weights
    python -m src.level1_radar.etci_reproduce download-data           # Hugging Face (about 5 GB)
    python -m src.level1_radar.etci_reproduce inspect                 # check the layout first
    python -m src.level1_radar.etci_reproduce fit-scaling --method val-search
    python scripts/evaluate.py --config configs/E0.yaml               # ETCI validation tiles
    python scripts/evaluate.py --config configs/E3.yaml               # Sen1Floods11
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
import yaml

from src.common.config import data_path, get_data_root, repo_root
from src.common.logging import get_logger
from src.level1_radar.models import make_network
from src.level1_radar.predict import _d4, _d4_inverse

log = get_logger("etci")
WEIGHTS_URL = (
    "https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/releases/"
    "download/v1.0.0/pretrained_weights.tar.gz"
)
HF_DATASET = "blanchon/ETCI-2021-Flood-Detection"


# --------------------------------------------------------------------------- encoding


def s1_to_rgb(vv01: np.ndarray, vh01: np.ndarray) -> np.ndarray:
    """ETCI's 3-channel input from VV and VH in [0, 1] (exactly as ``etci_dataset.s1_to_rgb``).

    Note: the original calls ``np.nan_to_num(x, 0)``, where 0 is the ``copy`` flag, so NaN
    becomes 0 and +inf becomes a huge number that the clip turns into 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(np.nan_to_num(vh01 / vv01), 0, 1)
    return np.stack((vv01, vh01, 1 - ratio)).astype(np.float32)


def load_scaling(spec: str | Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """dB range per band mapped linearly to [0, 1]: ``{"VV": (lo, hi), "VH": (lo, hi)}``.

    ``spec`` is a dict or a YAML path relative to the repo (``configs/etci_scaling.yaml``).
    """
    data = (
        spec
        if isinstance(spec, Mapping)
        else yaml.safe_load((repo_root() / spec).read_text(encoding="utf-8"))
    )
    return {b: (float(data["db_range"][b][0]), float(data["db_range"][b][1])) for b in ("VV", "VH")}


def db_to_unit(db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map of dB to [0, 1] with clipping; NaN (no data) becomes 0 like a black PNG pixel."""
    out = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def quantize_like_png(x01: np.ndarray) -> np.ndarray:
    """Round to 8-bit steps, as a PNG would store it."""
    return (np.round(x01 * 255.0) / 255.0).astype(np.float32)


# --------------------------------------------------------------------------- ensemble


class EtciEnsemble:
    """The released ETCI models; ``predict_*`` return p(flood) per pixel."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        device: torch.device,
        tta: bool,
        batch_size: int = 8,
    ) -> None:
        self.models = [m.eval().to(device) for m in models]
        self.device, self.tta, self.batch_size = device, tta, batch_size

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> EtciEnsemble:
        """``cfg``: ``weights_dir`` (relative to DATA_ROOT), ``members`` [{arch, file}], ``tta``."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        for member in cfg["members"]:
            net = make_network(member["arch"], "mobilenet_v2", in_channels=3, classes=2)
            path = data_path(cfg["weights_dir"], member["file"])
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            models.append(net)
        return cls(models, device, tta=bool(cfg.get("tta", True)))

    @torch.no_grad()
    def predict_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """p(flood) for ``(3, H, W)`` or ``(N, 3, H, W)`` encoded inputs (H, W divisible by 32)."""
        single = rgb.ndim == 3
        batch = torch.from_numpy(rgb[None] if single else rgb).to(self.device)
        variants = [(k, f) for k in range(4) for f in (False, True)] if self.tta else [(0, False)]
        logits = torch.zeros(batch.shape[0], 2, *batch.shape[-2:], device=self.device)
        for model in self.models:
            for k, flip in variants:
                logits += _d4_inverse(model(_d4(batch, k, flip)), k, flip)
        prob = torch.softmax(logits / (len(self.models) * len(variants)), dim=1)[:, 1]
        out = prob.cpu().numpy().astype(np.float32)
        return out[0] if single else out

    def predict_png(self, vv01: np.ndarray, vh01: np.ndarray) -> np.ndarray:
        """p(flood) from ETCI PNG tiles already divided by 255."""
        return self.predict_rgb(s1_to_rgb(vv01, vh01))

    def predict_db(
        self, vv_db: np.ndarray, vh_db: np.ndarray, scaling: Mapping[str, tuple[float, float]]
    ) -> np.ndarray:
        """p(flood) from Sentinel-1 dB values via the fitted dB -> PNG mapping (E3)."""
        vv01 = quantize_like_png(db_to_unit(vv_db, *scaling["VV"]))
        vh01 = quantize_like_png(db_to_unit(vh_db, *scaling["VH"]))
        return self.predict_png(vv01, vh01)


# --------------------------------------------------------------------------- ETCI tiles


def read_png(path: str | Path) -> np.ndarray:
    """First band of a PNG as uint8 ``(H, W)`` (grayscale or RGB with equal bands)."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def tiles_from_folders(root: Path) -> pd.DataFrame:
    """Rows for the original layout ``<region>_<date>/tiles/{vv,vh,flood_label,...}/*.png``."""
    rows = []
    for vv in sorted(root.rglob("tiles/vv/*_vv.png")):
        tiles = vv.parent.parent
        stem = vv.name.removesuffix("_vv.png")
        region_date = tiles.parent.name
        flood = tiles / "flood_label" / f"{stem}.png"
        rows.append(
            {
                "chip_id": stem,
                "event": region_date.split("_")[0],
                "vv_png": str(vv),
                "vh_png": str(tiles / "vh" / f"{stem}_vh.png"),
                "label_png": str(flood) if flood.exists() else None,
                "water_body_png": str(tiles / "water_body_label" / f"{stem}.png"),
            }
        )
    return pd.DataFrame(rows)


def _decode(cell: Any) -> np.ndarray:
    """A parquet image cell (bytes, or a {"bytes": ...} struct) as uint8 ``(H, W)``."""
    from PIL import Image

    raw = cell["bytes"] if isinstance(cell, Mapping) else cell
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("L"))


def tiles_from_parquet(root: Path, split: str) -> pd.DataFrame:
    """Rows from Hugging Face parquet shards whose path contains ``split``.

    Column names are matched loosely (``vv``, ``vh``, ``flood``, ``water``, ``region``/``event``);
    image cells are decoded on demand by :func:`etci_row_arrays`.
    """
    shards = sorted(p for p in root.rglob("*.parquet") if split in str(p.relative_to(root)))
    if not shards:
        raise FileNotFoundError(f"no parquet files for split {split!r} under {root}")
    frames = []
    for shard in shards:
        df = pd.read_parquet(shard)
        cols = {c.lower(): c for c in df.columns}

        def pick(*keys: str, df_cols: dict = cols) -> str | None:
            for key in keys:
                for low, orig in df_cols.items():
                    if key in low:
                        return orig
            return None

        out = pd.DataFrame(
            {
                "vv_cell": df[pick("vv")],
                "vh_cell": df[pick("vh")],
                "label_cell": df[pick("flood")] if pick("flood") else None,
            }
        )
        region = pick("region", "event", "location")
        out["event"] = df[region].astype(str) if region else shard.parent.name
        out["chip_id"] = [f"{shard.stem}_{i}" for i in range(len(df))]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def etci_rows(split: str = "validation") -> pd.DataFrame:
    """Labelled ETCI tiles of a split from ``$DATA_ROOT/etci2021`` (either layout)."""
    root = get_data_root() / "etci2021"
    folder_split = {"validation": "test", "train": "train"}.get(split, split)
    if (root / folder_split).is_dir() and any((root / folder_split).rglob("*_vv.png")):
        rows = tiles_from_folders(root / folder_split)
        return rows[rows["label_png"].notna()].reset_index(drop=True)
    return tiles_from_parquet(root, split)


def etci_row_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """VV and VH in [0, 1] and the flood label (0/1 int16) of one ETCI row."""
    if "vv_png" in row:
        vv, vh = read_png(row["vv_png"]), read_png(row["vh_png"])
        label = read_png(row["label_png"]) if row.get("label_png") else None
    else:
        vv, vh = _decode(row["vv_cell"]), _decode(row["vh_cell"])
        label = _decode(row["label_cell"]) if row.get("label_cell") is not None else None
    label01 = None if label is None else (label > 127).astype(np.int16)
    return vv / 255.0, vh / 255.0, label01


def is_blank(vv01: np.ndarray) -> bool:
    """ETCI's ``filter_df`` rule: tiles whose VV is only 0 and/or 255 carry no data."""
    return bool(np.isin(np.unique(np.round(vv01 * 255)), [0, 255]).all())


# --------------------------------------------------------------------------- command line

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("download-weights")
def download_weights(
    out: str = typer.Option("weights/etci", help="Relative to $DATA_ROOT"),
) -> None:
    """Fetch release v1.0.0 weights (108 MB; an uncompressed tar despite the .tar.gz name)."""
    dest = data_path(out)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "pretrained_weights.tar"
    if not archive.exists():
        log.info("downloading %s", WEIGHTS_URL)
        urllib.request.urlretrieve(WEIGHTS_URL, archive)
    with tarfile.open(archive) as tar:  # mode "r:*" detects the real format
        tar.extractall(dest, filter="data")
    log.info("weights in %s: %s", dest, sorted(p.name for p in dest.glob("*.pth")))


@app.command("download-data")
def download_data(repo_id: str = typer.Option(HF_DATASET)) -> None:
    """Download the ETCI 2021 dataset from Hugging Face into ``$DATA_ROOT/etci2021``."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(get_data_root() / "etci2021")
    )
    log.info("downloaded to %s", path)


@app.command("inspect")
def inspect(split: str = typer.Option("validation"), n: int = typer.Option(3)) -> None:
    """Print the layout, counts and value ranges (open a sample before writing any loader)."""
    root = get_data_root() / "etci2021"
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(
        f"{len(files)} files under {root}; first: {[str(p.relative_to(root)) for p in files[:8]]}"
    )
    rows = etci_rows(split)
    print(
        f"{len(rows)} labelled tiles in split {split!r}; events: "
        f"{rows['event'].value_counts().to_dict()}"
    )
    for row in rows.head(n).to_dict("records"):
        vv, vh, label = etci_row_arrays(row)
        print(
            row["chip_id"],
            vv.shape,
            f"VV {vv.min():.3f}-{vv.max():.3f}",
            f"VH {vh.min():.3f}-{vh.max():.3f}",
            f"label values {None if label is None else np.unique(label).tolist()}",
            f"blank={is_blank(vv)}",
        )


@app.command("fit-scaling")
def fit_scaling(
    method: str = typer.Option("val-search", help="val-search (Sen1Floods11 val IoU) or quantile"),
    config: Path = typer.Option(Path("configs/E3.yaml")),
    chips: int = typer.Option(60, help="Validation chips used by val-search."),
    out: Path = typer.Option(None, help="Default: configs/etci_scaling.yaml"),
) -> None:
    """Estimate the dB -> [0, 1] mapping and write ``configs/etci_scaling.yaml``."""
    from src.common.config import load_config
    from src.common.metrics import SplitScorer, binarize
    from src.level1_radar.datasets import load_image, load_label
    from src.level1_radar.s11 import read_split

    cfg = load_config(config)
    out = out or repo_root() / "configs" / "etci_scaling.yaml"
    if method == "quantile":
        result = _quantile_fit()
    elif method == "val-search":
        etci_cfg = {**cfg["etci"], "tta": False}
        ensemble = EtciEnsemble.from_config(etci_cfg)
        rows = read_split("s11_val").head(chips).to_dict("records")
        data = [(load_image(r, ["VV", "VH"]), load_label(r["label"]), r["event"]) for r in rows]

        def objective(scale: dict[str, tuple[float, float]]) -> float:
            scorer = SplitScorer()
            for image, label, event in data:
                prob = ensemble.predict_db(image[0], image[1], scale)
                scorer.update(binarize(prob), label, event=event)
            return scorer.overall.iou

        result = _coordinate_search(objective, start={"VV": (-30.0, 5.0), "VH": (-35.0, 0.0)})
    else:
        raise typer.BadParameter("method must be val-search or quantile")
    result["fitted_on"] = method
    out.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(out.read_text())


def _coordinate_search(
    objective: Any, start: dict[str, tuple[float, float]], steps: Sequence[float] = (5.0, 2.5, 1.0)
) -> dict[str, Any]:
    """Greedy search over (lo, hi) of each band with shrinking step sizes."""
    best = {b: list(v) for b, v in start.items()}
    best_score = objective({b: tuple(v) for b, v in best.items()})
    log.info("start %s IoU %.4f", best, best_score)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for band in ("VV", "VH"):
                for idx in (0, 1):
                    for delta in (-step, step):
                        cand = {b: list(v) for b, v in best.items()}
                        cand[band][idx] += delta
                        if cand[band][1] - cand[band][0] < 5:
                            continue
                        score = objective({b: tuple(v) for b, v in cand.items()})
                        if score > best_score + 1e-4:
                            best, best_score, improved = cand, score, True
                            log.info("step %.1f: %s IoU %.4f", step, best, best_score)
    return {
        "db_range": {b: [float(v[0]), float(v[1])] for b, v in best.items()},
        "val_iou": float(best_score),
    }


def _quantile_fit(
    quantiles: Sequence[float] = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98), max_tiles: int = 2000
) -> dict[str, Any]:
    """Match value quantiles of ETCI training PNGs and Sen1Floods11 training dB per band."""
    from src.level1_radar.datasets import load_image
    from src.level1_radar.s11 import read_split

    rng = np.random.default_rng(0)
    rows = etci_rows("train")
    rows = rows.iloc[rng.permutation(len(rows))[:max_tiles]]
    png = {"VV": [], "VH": []}
    for row in rows.to_dict("records"):
        vv, vh, _ = etci_row_arrays(row)
        if is_blank(vv):
            continue
        png["VV"].append(rng.choice(vv.ravel(), 2000))
        png["VH"].append(rng.choice(vh.ravel(), 2000))
    db = {"VV": [], "VH": []}
    for row in read_split("s11_train").to_dict("records"):
        image = load_image(row, ["VV", "VH"])
        for i, band in enumerate(("VV", "VH")):
            values = image[i][np.isfinite(image[i])]
            db[band].append(rng.choice(values, 2000))
    result = {}
    for band in ("VV", "VH"):
        x = np.quantile(np.concatenate(db[band]), quantiles)
        y = np.quantile(np.concatenate(png[band]), quantiles)
        slope, intercept = np.polyfit(x, y, 1)
        result[band] = [float(-intercept / slope), float((1 - intercept) / slope)]
    return {"db_range": result}


if __name__ == "__main__":
    app()
