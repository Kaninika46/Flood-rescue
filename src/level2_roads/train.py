"""Train the SpaceNet 8 foundation or flood network (S0-S3). Run in the "sn8" environment.

    python -m src.level2_roads.train --config configs/S0.yaml --task foundation --seed 0
    python -m src.level2_roads.train --config configs/S0.yaml --task flood --seed 0
    python -m src.level2_roads.train --config configs/S0.yaml --task flood --resume
    python -m src.level2_roads.train --config configs/S0.yaml --task foundation --smoke

Adapted from ``third_party/spacenet8/baseline/train_foundation_features.py`` and
``train_flood.py`` (Apache-2.0, commit 30ca30c). Unchanged: the models (ResNet-34 U-Net with
outputs [1 building, 8 road speed]; Siamese ResNet-34 with 5 flood classes), the losses
(foundation: 0.5 x (0.75 focal + 0.25 soft Dice on road speed) + 0.5 x BCE on buildings;
flood: cross-entropy), Adam, StepLR(40, 0.5), best checkpoint by validation loss.
Added (hardware rules, README "Project rules"): mixed precision, random crops (``data.crop``),
gradient accumulation, seeds, a checkpoint every epoch, ``--resume``, ``--smoke``, tracking through
``src/common/logging.py``, pixel IoU metrics on validation, optional initial weights
(``model.init_weights``, used by S2) and ``model.pretrained: false`` for offline smoke tests.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from torch import nn
from torch.utils.data import DataLoader

from src.common import gpu
from src.common.checkpoint import load_checkpoint
from src.common.config import data_path, effective_batch_size, load_config, run_name
from src.common.logging import get_logger, start_run
from src.common.metrics import SplitScorer
from src.common.seed import seed_everything, seed_worker, torch_generator
from src.level2_roads.data import read_sn8_split
from src.level2_roads.dataset import SN8Dataset

log = get_logger("sn8_train")
app = typer.Typer(add_completion=False, help=__doc__)
TASKS = ("foundation", "flood")


def build_model(task: str, model_cfg: Mapping[str, Any]) -> nn.Module:
    """Baseline network for the task; ImageNet encoder unless ``pretrained: false``."""
    from src.level2_roads.sn8.models import abstract_model, unet

    if not model_cfg.get("pretrained", True):
        abstract_model.AbstractModel.initialize_encoder = lambda self, model, url: None
    if task == "foundation":
        model = unet.Resnet34_upsample(num_classes=[1, 8], num_channels=3)
    else:
        model = unet.Resnet34_siamese_upsample(num_classes=5, num_channels=3)
    init = model_cfg.get("init_weights", {}).get(task) if model_cfg.get("init_weights") else None
    if init:
        # plain tensors only (scripts/prepare_sn5_weights.py converts the SN5 pickle safely)
        state = torch.load(data_path(init), map_location="cpu", weights_only=True)
        state = state.get("model", state)
        own = model.state_dict()
        matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(matched)
        model.load_state_dict(own)
        log.info("initialised %d / %d tensors from %s", len(matched), len(own), init)
    return model


def foundation_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
) -> torch.Tensor:
    """Baseline foundation loss, computed in float32."""
    from src.level2_roads.sn8.losses import focal, soft_dice_loss

    building_pred, road_pred = (o.float() for o in outputs)
    bce = F.binary_cross_entropy_with_logits(building_pred, batch["building"])
    y = torch.sigmoid(road_pred)
    road = cfg["focal_weight"] * focal(y, batch["roadspeed"]) + cfg["dice_weight"] * soft_dice_loss(
        y, batch["roadspeed"]
    )
    return cfg["road_weight"] * road + cfg["building_weight"] * bce


def forward(model: nn.Module, task: str, batch: Mapping[str, torch.Tensor]) -> Any:
    """Model outputs for a batch."""
    if task == "foundation":
        return model(batch["preimg"])
    return model(batch["preimg"], batch["postimg"])


def compute_loss(
    task: str, outputs: Any, batch: Mapping[str, torch.Tensor], loss_cfg: Mapping[str, Any]
) -> torch.Tensor:
    """Task loss (float32)."""
    if task == "foundation":
        return foundation_loss(outputs, batch, loss_cfg)
    return F.cross_entropy(outputs.float(), batch["flood"])


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors to the device (non-tensors stay)."""
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def update_scorers(
    task: str, outputs: Any, batch: Mapping[str, Any], scorers: dict[str, SplitScorer]
) -> None:
    """Pixel IoU: building and road (foundation); flooded building / road (flood)."""
    events = list(batch["aoi"])
    if task == "foundation":
        building = (torch.sigmoid(outputs[0].float())[:, 0] >= 0.5).cpu().numpy().astype(np.uint8)
        road = (torch.sigmoid(outputs[1].float())[:, -1] >= 0.5).cpu().numpy().astype(np.uint8)
        scorers["building"].update(
            building, batch["building"][:, 0].cpu().numpy().astype(np.int16), event=events
        )
        scorers["road"].update(
            road, batch["roadspeed"][:, -1].cpu().numpy().astype(np.int16), event=events
        )
    else:
        pred = outputs.float().argmax(1).cpu().numpy()
        ref = batch["flood"].cpu().numpy()
        for name, cls in (("flooded_building", 2), ("flooded_road", 4)):
            scorers[name].update(
                (pred == cls).astype(np.uint8), (ref == cls).astype(np.int16), event=events
            )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
    use_amp: bool,
    optimizer: Any = None,
    scaler: Any = None,
    accum: int = 1,
    max_batches: int = 0,
    scorers: dict[str, SplitScorer] | None = None,
) -> float:
    """Train (optimizer given) or validate; returns the mean loss per batch."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    n = len(loader) if not max_batches else min(max_batches, len(loader))
    if training:
        optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if i >= n:
            break
        batch = to_device(batch, device)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, dtype=torch.float16, enabled=use_amp),
        ):
            outputs = forward(model, task, batch)
        with torch.set_grad_enabled(training):
            loss = compute_loss(task, outputs, batch, loss_cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {loss.item()} at batch {i}")
        if training:
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == n:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        elif scorers is not None:
            update_scorers(task, outputs, batch, scorers)
        total += loss.item()
        count += 1
    return total / max(1, count)


@app.command()
def main(
    config: Path = typer.Option(..., help="configs/S0.yaml"),
    task: str = typer.Option(..., help="foundation or flood"),
    seed: int = typer.Option(0),
    set_: list[str] = typer.Option([], "--set", help="Config override key.sub=value."),
    resume: bool = typer.Option(False, help="Continue from last.pt."),
    smoke: bool = typer.Option(False, help="2 train + 2 val batches, 1 epoch."),
    device_name: str = typer.Option(None, "--device"),
) -> None:
    """Train one network of one run."""
    if task not in TASKS:
        raise typer.BadParameter(f"task must be one of {TASKS}")
    cfg = load_config(config, set_)
    seed_everything(seed)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    name = f"{run_name(cfg['experiment'], seed)}-{task}" + ("-smoke" if smoke else "")
    out = data_path("checkpoints", name)
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"
    state = (
        load_checkpoint(last_path, resume=True)
        if (resume and last_path.exists())
        else None
    )
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    batch_size, accum = int(tcfg["batch_size"]), int(tcfg.get("accum_steps", 1))
    if effective_batch_size(cfg) != batch_size * accum:
        raise ValueError("effective batch size mismatch")

    train_rows, val_rows = read_sn8_split(dcfg["train_split"]), read_sn8_split(dcfg["val_split"])
    if smoke:
        present = [data_path(p).exists() for p in train_rows["preimg"]]
        train_rows = train_rows[present].head(batch_size * 2)
        val_rows = val_rows[[data_path(p).exists() for p in val_rows["preimg"]]].head(2)
        if val_rows.empty:
            val_rows = train_rows.head(1)
    workers = 0 if smoke else int(tcfg.get("num_workers", 4))
    train_ds = SN8Dataset(train_rows, task, crop=int(dcfg.get("crop", 512)))
    val_ds = SN8Dataset(val_rows, task, crop=dcfg.get("val_crop"))
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers)

    model = build_model(task, mcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(tcfg.get("lr_step", 40)), gamma=float(tcfg.get("lr_gamma", 0.5))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch, best_loss = 0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_loss = state["epoch"] + 1, state["best_loss"]
        log.info("resuming %s at epoch %d", name, start_epoch)
    epochs = 1 if smoke else int(tcfg["epochs"])
    loss_cfg = tcfg.get("loss", {})
    meta = {"task": task, "model_cfg": dict(mcfg), "experiment": cfg["experiment"], "seed": seed}

    with start_run(
        cfg,
        seed,
        job_type=f"train_{task}",
        smoke=smoke,
        tags=[task],
        resume=state.get("tracker") if state else None,
        save_config_to=out / "config.yaml",
    ) as run:
        for epoch in range(start_epoch, epochs):
            start = time.perf_counter()
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                worker_init_fn=seed_worker,
                generator=torch_generator(seed * 1000 + epoch),
            )
            train_loss = run_epoch(
                model,
                loader,
                task,
                device,
                loss_cfg,
                use_amp,
                optimizer,
                scaler,
                accum,
                2 if smoke else 0,
            )
            scheduler.step()
            names = (
                ("building", "road")
                if task == "foundation"
                else ("flooded_building", "flooded_road")
            )
            scorers = {n: SplitScorer() for n in names}
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    task,
                    device,
                    loss_cfg,
                    use_amp,
                    max_batches=2 if smoke else 0,
                    scorers=scorers,
                )
            metrics = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
                "epoch_min": (time.perf_counter() - start) / 60,
                "gpu_peak_memory_gb": gpu.peak_memory_gb() or 0.0,
            }
            for n, sc in scorers.items():
                metrics[f"val/{n}_iou"] = sc.overall.iou
                metrics[f"val/{n}_recall"] = sc.overall.recall
            run.log_metrics(metrics, step=epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {**meta, "model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                    best_path,
                )
            tmp = last_path.with_suffix(".tmp")
            torch.save(
                {
                    **meta,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "tracker": run.resume_info(),
                },
                tmp,
            )
            tmp.replace(last_path)
            log.info(
                "epoch %d: train %.4f val %.4f (best %.4f)", epoch, train_loss, val_loss, best_loss
            )
        run.log_summary({"val/best_loss": best_loss})


if __name__ == "__main__":
    app()
