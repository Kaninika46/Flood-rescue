"""Our checkpoints load with weights_only=True (src/common/checkpoint.py); tiny CPU models only."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

from src.common.checkpoint import load_checkpoint


def tiny() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1))


def level1_last(model: nn.Module) -> dict:
    """Shaped like level1_radar.train's last.pt: AdamW, LambdaLR, GradScaler, RNG state."""
    from src.level1_radar.train import rng_state

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.zeros(1, 3, 8, 8)).sum().backward()
    opt.step()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return {
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
        "epoch": 3, "best_iou": 0.61, "bad_epochs": 1, "channels": ["VV", "VH", "HAND"],
        "model_cfg": {"arch": "unet", "encoder": "resnet34"}, "norm": {"VV": [-10.0, 4.0]},
        "experiment": "E6", "seed": 0, "tracker": {"backend": "mlflow", "id": "abc"},
    }


def test_best_loads_without_any_allowlist(tmp_path: Path) -> None:
    model = tiny()
    torch.save({"model": model.state_dict(), "channels": ["VV", "VH"], "val_iou": 0.6,
                "model_cfg": {"arch": "unet"}, "epoch": 2}, tmp_path / "best.pt")
    ckpt = load_checkpoint(tmp_path / "best.pt")
    again = tiny()
    again.load_state_dict(ckpt["model"])
    assert ckpt["channels"] == ["VV", "VH"]


def test_level1_resume_state_round_trips(tmp_path: Path) -> None:
    import numpy as np

    saved = level1_last(tiny())
    torch.save(saved, tmp_path / "last.pt")
    ckpt = load_checkpoint(tmp_path / "last.pt", resume=True)
    assert ckpt["epoch"] == 3
    np.testing.assert_array_equal(ckpt["rng"]["numpy"][1], saved["rng"]["numpy"][1])
    opt = torch.optim.AdamW(tiny().parameters(), lr=1e-3)
    opt.load_state_dict(ckpt["optimizer"])


def test_numpy_types_are_allowed_only_for_resume(tmp_path: Path) -> None:
    torch.save(level1_last(tiny()), tmp_path / "last.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(tmp_path / "last.pt")  # not a resume load: no NumPy allowlist


def test_level2_checkpoints_load(tmp_path: Path) -> None:
    """Shaped like level2_roads.train's best.pt / last.pt (Adam, StepLR, GradScaler)."""
    model = tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
    meta = {"task": "foundation", "model_cfg": {"encoder": "resnet34"}, "experiment": "S0",
            "seed": 0}
    torch.save({**meta, "model": model.state_dict(), "epoch": 1, "val_loss": 0.3},
               tmp_path / "best.pt")
    torch.save({**meta, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": torch.amp.GradScaler("cpu", enabled=False).state_dict(),
                "epoch": 1, "best_loss": 0.3, "tracker": {"backend": "mlflow", "id": "x"}},
               tmp_path / "last.pt")
    assert load_checkpoint(tmp_path / "best.pt")["task"] == "foundation"
    assert load_checkpoint(tmp_path / "last.pt", resume=True)["best_loss"] == 0.3


class _Booby:
    """A pickle that would create a file when unpickled with the full (unsafe) loader."""

    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (open, (str(self.target), "w"))


@pytest.mark.parametrize("resume", [False, True])
def test_tampered_checkpoint_does_not_run_code(tmp_path: Path, resume: bool) -> None:
    marker = tmp_path / "pwned"
    torch.save({"model": {}, "payload": _Booby(marker)}, tmp_path / "evil.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(tmp_path / "evil.pt", resume=resume)
    assert not marker.exists()
