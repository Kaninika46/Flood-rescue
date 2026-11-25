"""Write SpaceNet 8 prediction GeoTIFFs in the baseline's formats (B3, B6).

Adapted from ``third_party/spacenet8/baseline/foundation_eval.py`` and ``flood_eval.py``
(Apache-2.0, commit 30ca30c). Same outputs, so the baseline post-processing runs unchanged:

Written to ``roadspeed/``, ``building/`` and ``flood/`` sub-folders (the baseline scripts read
every .tif in a folder):

* ``<pre>_roadspeedpred.tif``: 8 bands uint8, sigmoid x 255 (band 8 = any road)
* ``<pre>_buildingpred.tif``: 1 band uint8, building 0/1
* ``<pre>_floodpred.tif``: 1 band uint8, argmax class (0 bg, 1 building, 2 flooded building,
  3 road, 4 flooded road)

Added for S3: with several flood checkpoints, the softmax outputs are averaged
(``<pre>_floodprob.tif``, 5 bands x 255) and ``<pre>_flooddisagree.tif`` stores the share of
members whose argmax differs from the ensemble's (0-255). Full tiles, mixed precision.

    python -m src.level2_roads.infer --config configs/S0.yaml --split sn8_val --out-dir DIR
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from src.common.config import data_path, load_config, run_name
from src.common.logging import get_logger
from src.level2_roads.data import read_sn8_split, write_like_pre
from src.level2_roads.dataset import SN8Dataset
from src.level2_roads.train import build_model

log = get_logger("sn8_infer")
app = typer.Typer(add_completion=False, help=__doc__)


def load_models(cfg: dict, task: str, seeds: Sequence[int], device: torch.device) -> list:
    """Best checkpoints of the task for the given seeds."""
    models = []
    for seed in seeds:
        path = data_path("checkpoints", f"{run_name(cfg['experiment'], seed)}-{task}", "best.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(task, {**cfg["model"], "pretrained": False, "init_weights": None})
        model.load_state_dict(ckpt["model"])
        models.append(model.eval().to(device))
    return models


@torch.no_grad()
def run_inference(
    cfg: dict, split: str, out_dir: Path, seeds: Sequence[int] | None = None, limit: int = 0
) -> Path:
    """Predict every tile of a split; returns ``out_dir``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    seeds = list(seeds or cfg.get("seeds", [0]))
    rows = read_sn8_split(split)
    rows = rows[[data_path(p).exists() for p in rows["preimg"]]]
    if limit:
        rows = rows.head(limit)
    dirs = {k: out_dir / k for k in ("roadspeed", "building", "flood")}
    for d in dirs.values():  # the baseline scripts read every .tif in their folder
        d.mkdir(parents=True, exist_ok=True)
    foundation = load_models(cfg, "foundation", seeds[:1], device)[0]
    floods = load_models(cfg, "flood", seeds, device)
    ds = SN8Dataset(rows, "flood", crop=None, with_targets=False)
    for i, item in enumerate(DataLoader(ds, batch_size=1, num_workers=2)):
        pre_path = data_path(item["preimg_path"][0])
        stem = pre_path.stem
        pre = item["preimg"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            building, roadspeed = foundation(pre)
        building = (torch.sigmoid(building.float())[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)
        roadspeed = (torch.sigmoid(roadspeed.float())[0] * 255).cpu().numpy().astype(np.uint8)
        write_like_pre(dirs["roadspeed"] / f"{stem}_roadspeedpred.tif", pre_path, roadspeed)
        write_like_pre(dirs["building"] / f"{stem}_buildingpred.tif", pre_path, building[None])
        post = item["postimg"].to(device)
        probs = []
        for model in floods:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(pre, post)
            probs.append(torch.softmax(logits.float(), dim=1)[0].cpu().numpy())
        stack = np.stack(probs)
        mean = stack.mean(axis=0)
        pred = mean.argmax(axis=0).astype(np.uint8)
        write_like_pre(dirs["flood"] / f"{stem}_floodpred.tif", pre_path, pred[None])
        if len(floods) > 1:
            (out_dir / "flood_ensemble").mkdir(exist_ok=True)
            disagree = (stack.argmax(axis=1) != pred[None]).mean(axis=0)
            write_like_pre(
                dirs["flood"].parent / "flood_ensemble" / f"{stem}_floodprob.tif",
                pre_path,
                (mean * 255).astype(np.uint8),
            )
            write_like_pre(
                dirs["flood"].parent / "flood_ensemble" / f"{stem}_flooddisagree.tif",
                pre_path,
                (disagree * 255).astype(np.uint8)[None],
            )
        if (i + 1) % 20 == 0:
            log.info("%d / %d tiles", i + 1, len(rows))
    return out_dir


@app.command()
def main(
    config: Path = typer.Option(...),
    split: str = typer.Option(...),
    out_dir: Path = typer.Option(...),
    seed: list[int] = typer.Option([]),
    limit: int = typer.Option(0),
) -> None:
    """Write prediction GeoTIFFs for a split."""
    run_inference(load_config(config), split, out_dir, seed or None, limit)


if __name__ == "__main__":
    app()
