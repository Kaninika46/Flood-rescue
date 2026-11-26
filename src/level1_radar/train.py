"""Train a radar flood U-Net on Sen1Floods11 (E4, E5, E6 and their variants).

    python -m src.level1_radar.train --config configs/E4.yaml --seed 0
    python -m src.level1_radar.train --config configs/E4.yaml --seed 0 --resume     # after a stop
    python -m src.level1_radar.train --config configs/E4.yaml --smoke               # 2 batches
    python -m src.level1_radar.train --config configs/E4.yaml --set train.batch_size=4 \\
        --set train.accum_steps=4                                                  # smaller GPU

What it does (hardware and reproducibility rules, README "Project rules"):

* mixed precision on CUDA, gradient accumulation (``train.batch_size x train.accum_steps``
  is the effective batch), random crops of ``data.crop`` pixels with flips and rotations
* Dice + focal loss that ignores label -1; AdamW; linear warm-up then cosine decay
* validation on full ``data.val_split`` chips after every epoch; early stopping on pooled
  flood IoU (``train.patience`` epochs without improvement)
* ``last.pt`` (full state, for ``--resume``) and ``best.pt`` (weights + channels + norm)
  in ``$DATA_ROOT/checkpoints/<experiment>-s<seed>/`` every epoch, written atomically
* with ``data.pseudo`` set, weak chips with pseudo-labels (``pseudo_label.py``) join training
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from src.common import gpu
from src.common.config import data_path, effective_batch_size, load_config, repo_root, run_name
from src.common.logging import get_logger, start_run
from src.common.metrics import SplitScorer, binarize
from src.common.seed import seed_everything, seed_worker, torch_generator
from src.level1_radar.datasets import Norm, Sen1Floods11Dataset
from src.level1_radar.losses import masked_dice_focal_loss
from src.level1_radar.models import build_model
from src.level1_radar.s11 import read_split

log = get_logger("train")
app = typer.Typer(add_completion=False, help=__doc__)


def pseudo_label_path(pseudo_dir: str, chip_id: str) -> str:
    """Relative path of one pseudo-label raster."""
    return f"{pseudo_dir}/{chip_id}_pseudo.tif"


def build_train_rows(data_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Hand-labelled training rows plus, if configured, weak chips with pseudo-labels.

    The returned frame has a ``target`` column with the label path to train on. Hand-labelled
    rows are repeated ``data.crops_per_chip`` times (one random crop each); weak chips once.
    """
    rows = read_split(data_cfg["train_split"]).copy()
    rows["target"] = rows["label"]
    rows["source"] = "hand"
    repeat = int(data_cfg.get("crops_per_chip", 1))
    if repeat > 1:  # several random crops of every hand-labelled chip per epoch
        rows = pd.concat([rows] * repeat, ignore_index=True)
    pseudo = data_cfg.get("pseudo")
    if not pseudo:
        return rows
    weak = read_split(pseudo.get("split", "s11_weak")).copy()
    weak = weak[~weak["event"].isin(pseudo.get("exclude_events", []))]
    weak["target"] = [pseudo_label_path(pseudo["dir"], c) for c in weak["chip_id"]]
    exists = [data_path(t).exists() for t in weak["target"]]
    if not any(exists):
        raise FileNotFoundError(
            f"no pseudo-labels in $DATA_ROOT/{pseudo['dir']}: run src.level1_radar.pseudo_label"
        )
    weak = weak[exists]
    if pseudo.get("max_chips"):
        weak = weak.sample(n=min(len(weak), pseudo["max_chips"]), random_state=0)
    weak["source"] = "pseudo"
    log.info("training on %d hand-labelled + %d pseudo-labelled chips", len(rows), len(weak))
    return pd.concat([rows, weak], ignore_index=True)  # weak rows have no label/jrc


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the base LR, then cosine decay to ``min_ratio x`` base LR."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_atomic(obj: Any, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename: a crash never leaves a broken file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state() -> dict[str, Any]:
    """Python, NumPy and torch RNG states (for exact-ish resume)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore what :func:`rng_state` saved."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
    accum_steps: int,
    use_amp: bool,
    max_batches: int = 0,
) -> float:
    """One pass over the training loader with gradient accumulation; returns the mean loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    n_batches = len(loader) if not max_batches else min(max_batches, len(loader))
    started = time.perf_counter()
    next_report = started + 15
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(image)
        loss = masked_dice_focal_loss(
            logits,
            label,
            dice_weight=loss_cfg.get("dice", 1.0),
            focal_weight=loss_cfg.get("focal", 1.0),
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {loss.item()} at batch {i}; stopping")
        scaler.scale(loss / accum_steps).backward()
        last_batch = i + 1 == n_batches
        if (i + 1) % accum_steps == 0 or last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total += loss.item() * image.shape[0]
        count += image.shape[0]
        now = time.perf_counter()
        if now >= next_report and not last_batch:
            batches_done = i + 1
            rate = batches_done / (now - started)
            log.info(
                "training batches %d/%d (%.1f%%), %.2f batches/s, ETA ~%.0fs",
                batches_done,
                n_batches,
                100 * batches_done / n_batches,
                rate,
                (n_batches - batches_done) / rate,
            )
            next_report = now + 15
    return total / max(1, count)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    max_batches: int = 0,
) -> SplitScorer:
    """Pooled flood IoU etc. on full validation chips."""
    model.eval()
    scorer = SplitScorer()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch["image"].to(device))
        prob = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        scorer.update(binarize(prob, threshold), batch["label"].numpy(), event=batch["event"])
    return scorer


@app.command()
def main(
    config: Path = typer.Option(..., help="Experiment config, e.g. configs/E4.yaml"),
    seed: int = typer.Option(0, help="Run seed (standard: 0, 1, 2)."),
    set_: list[str] = typer.Option([], "--set", help="Config override key.sub=value."),
    resume: bool = typer.Option(False, help="Continue from last.pt of this run."),
    smoke: bool = typer.Option(False, help="2 train + 2 val batches, 1 epoch, end to end."),
    out_dir: Path = typer.Option(None, help="Default: $DATA_ROOT/checkpoints/<run name>"),
    device_name: str = typer.Option(None, "--device", help="cuda or cpu (default: auto)."),
) -> None:
    """Train one run of an experiment."""
    cfg = load_config(config, set_)
    seed_everything(seed)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    name = run_name(cfg["experiment"], seed) + ("-smoke" if smoke else "")
    out = out_dir or data_path("checkpoints", name)
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"
    state = None
    if resume:
        if last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            log.info("resuming %s from epoch %d", name, state["epoch"] + 1)
        else:
            log.warning("--resume given but %s does not exist; starting from scratch", last_path)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    channels = list(data_cfg["channels"])
    norm = Norm.from_yaml(repo_root() / data_cfg["norm"])
    batch_size, accum = int(train_cfg["batch_size"]), int(train_cfg.get("accum_steps", 1))

    train_rows = build_train_rows(data_cfg)
    val_rows = read_split(data_cfg["val_split"])
    if smoke:
        train_rows, val_rows = train_rows.head(batch_size * 2), val_rows.head(2)
    workers = 0 if smoke else int(train_cfg.get("num_workers", 4))
    train_ds = Sen1Floods11Dataset(
        train_rows,
        channels,
        norm,
        train=True,
        crop=int(data_cfg.get("crop", 256)),
        label_column="target",
    )
    val_ds = Sen1Floods11Dataset(val_rows, channels, norm, train=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_cfg, channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = 1 if smoke else int(train_cfg["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / batch_size) / accum)
    scheduler = make_scheduler(
        optimizer,
        epochs * steps_per_epoch,
        int(train_cfg.get("warmup_epochs", 1)) * steps_per_epoch,
        float(train_cfg.get("min_lr_ratio", 0.01)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch, best_iou, bad_epochs = 0, -1.0, 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_iou, bad_epochs = (
            state["epoch"] + 1,
            state["best_iou"],
            state["bad_epochs"],
        )
        set_rng_state(state["rng"])

    patience = int(train_cfg.get("patience", 10))
    threshold = float(train_cfg.get("threshold", 0.5))
    loss_cfg = train_cfg.get("loss", {})
    max_batches = 2 if smoke else 0
    meta = {
        "model_cfg": dict(model_cfg),
        "channels": channels,
        "norm": norm.to_dict(),
        "experiment": cfg["experiment"],
        "seed": seed,
    }

    with start_run(
        cfg,
        seed,
        job_type="train",
        smoke=smoke,
        resume=state.get("tracker") if state else None,
        save_config_to=out / "config.yaml",
    ) as run:
        run.log_summary(
            {
                "train_chips": len(train_ds),
                "val_chips": len(val_ds),
                "effective_batch_size_check": batch_size * accum,
            }
        )
        if effective_batch_size(cfg) != batch_size * accum:
            raise ValueError("effective batch size mismatch")
        epoch = start_epoch - 1  # stays so if a resumed run has nothing left to do
        for epoch in range(start_epoch, epochs):
            log.info("starting epoch %d/%d", epoch + 1, epochs)
            start = time.perf_counter()
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=seed_worker,
                generator=torch_generator(seed * 1000 + epoch),
            )
            train_loss = train_one_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                scaler,
                device,
                loss_cfg,
                accum,
                use_amp,
                max_batches,
            )
            scorer = validate(model, val_loader, device, use_amp, threshold, max_batches)
            val = scorer.overall
            improved = val.iou > best_iou
            if improved:
                best_iou, bad_epochs = val.iou, 0
                save_atomic(
                    {**meta, "model": model.state_dict(), "epoch": epoch, "val_iou": val.iou},
                    best_path,
                )
            else:
                bad_epochs += 1
            run.log_metrics(
                {
                    "train/loss": train_loss,
                    "val/iou": val.iou,
                    "val/f1": val.f1,
                    "val/precision": val.precision,
                    "val/recall": val.recall,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_min": (time.perf_counter() - start) / 60,
                    "gpu_peak_memory_gb": gpu.peak_memory_gb() or 0.0,
                },
                step=epoch,
            )
            save_atomic(
                {
                    **meta,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "bad_epochs": bad_epochs,
                    "rng": rng_state(),
                    "tracker": run.resume_info(),
                },
                last_path,
            )
            log.info(
                "epoch %d/%d complete: loss %.4f val IoU %.4f (best %.4f), %.1fs%s",
                epoch + 1,
                epochs,
                train_loss,
                val.iou,
                best_iou,
                time.perf_counter() - start,
                " *" if improved else "",
            )
            if bad_epochs >= patience:
                log.info("early stop: no improvement for %d epochs", patience)
                break
        run.log_summary({"val/best_iou": best_iou, "epochs_run": epoch + 1})
    log.info("best checkpoint: %s (val IoU %.4f)", best_path, best_iou)


if __name__ == "__main__":
    app()
