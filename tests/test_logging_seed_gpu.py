"""Tests for src/common/logging.py, seed.py and gpu.py (no network, no GPU needed)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.common import gpu
from src.common.logging import (
    MlflowRun,
    choose_backend,
    flatten,
    git_info,
    run_metadata,
    start_run,
)
from src.common.seed import seed_everything

CFG = {"experiment": "E4", "train": {"batch_size": 4, "accum_steps": 4, "lr": 1e-3}}


def test_flatten_nested_config() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": {}}) == {"a.b": 1, "a.c.d": 2, "e": {}}


def test_run_metadata_has_everything_the_rules_require() -> None:
    meta = run_metadata(CFG, seed=1)
    assert meta["run_name"] == "E4-s1"
    assert meta["effective_batch_size"] == 16
    for key in ("git_commit", "git_dirty", "gpu_name", "gpu_total_memory_gb", "python_version"):
        assert key in meta
    assert isinstance(git_info()["git_commit"], str)


def test_backend_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOOD_TRACKER", raising=False)
    assert choose_backend("console") == "console"
    assert choose_backend(None, smoke=True) == "console"
    monkeypatch.setenv("FLOOD_TRACKER", "mlflow")
    assert choose_backend(None, smoke=True) == "mlflow"
    with pytest.raises(ValueError, match="unknown tracker"):
        choose_backend("tensorboard")
    monkeypatch.setenv("FLOOD_TRACKER", "")
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert choose_backend(None) == "mlflow"


def test_console_run_logs_runtime_and_status(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLOOD_TRACKER", raising=False)
    caplog.set_level("INFO", logger="flood")
    with start_run(CFG, seed=0, smoke=True) as run:
        run.log_metrics({"train/loss": 0.5, "skip": None}, step=0)
        run.log_summary({"val/iou": float("nan")})
    assert run.name == "E4-s0-smoke"
    assert "runtime_min" in caplog.text
    assert run.resume_info() == {"backend": "console", "id": "E4-s0-smoke"}


def test_failed_run_is_marked(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="flood")
    with pytest.raises(RuntimeError), start_run(CFG, seed=0, backend="console"):
        raise RuntimeError("boom")
    assert "(failed)" in caplog.text


def test_mlflow_offline_run_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    root = tmp_path / "mlruns"
    meta = run_metadata(CFG, seed=2)
    run = MlflowRun("E4-s2", meta, CFG, "train", ["test"], resume_id=None, root=root)
    run.log_metrics({"val/iou": 0.5, "val/event/Sri Lanka/iou": 0.25}, step=0)
    run.finish("interrupted")
    resumed = MlflowRun("E4-s2", meta, CFG, "train", [], resume_id=run.id, root=root)
    resumed.log_metrics({"val/iou": 0.6}, step=1)
    resumed.finish("ok")
    data = mlflow.get_run(run.id).data
    assert data.metrics["val/iou"] == 0.6
    assert data.params["train.batch_size"] == "4"
    assert data.params["meta.run_name"] == "E4-s2"
    assert data.tags["status"] == "ok"
    assert (root / "mlflow.db").is_file()


def test_seed_everything_repeats_numpy_and_random() -> None:
    seed_everything(1)
    first = (random.random(), np.random.rand())  # noqa: NPY002
    seed_everything(1)
    assert (random.random(), np.random.rand()) == first  # noqa: NPY002
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_everything_repeats_torch() -> None:
    torch = pytest.importorskip("torch")
    seed_everything(0)
    a = torch.rand(3)
    seed_everything(0)
    assert torch.equal(a, torch.rand(3))


def test_gpu_info_without_gpu_is_safe() -> None:
    info = gpu.gpu_info()
    assert isinstance(info["gpu_name"], str)
    if not gpu.cuda_available():
        assert info["gpu_name"] == "cpu"
        assert gpu.peak_memory_gb() is None
        assert gpu.describe(info) == "no CUDA GPU (CPU only)"
