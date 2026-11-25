"""GPU information for logs: name, total memory and peak memory used by this process.

Every run logs :func:`gpu_info` at the start and :func:`peak_memory_gb` at the end, so runs
on different laptops can be compared. Works without a GPU (reports ``"cpu"``).
"""

from __future__ import annotations

import contextlib
from typing import Any

_GB = 1024**3


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    """Return True if PyTorch is installed and sees a CUDA GPU."""
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def gpu_info(device: int = 0) -> dict[str, Any]:
    """Describe the GPU (or CPU) this run uses.

    Returns:
        A flat dict that is safe to log: ``gpu_name``, ``gpu_total_memory_gb``,
        ``gpu_compute_capability``, ``gpu_count``, ``cuda_version``, ``cudnn_version``,
        ``torch_version``. Missing values are ``None``.
    """
    torch = _torch()
    info: dict[str, Any] = {
        "gpu_name": "cpu",
        "gpu_total_memory_gb": None,
        "gpu_compute_capability": None,
        "gpu_count": 0,
        "cuda_version": None,
        "cudnn_version": None,
        "torch_version": None,
    }
    if torch is None:
        return info
    # CUDA/cuDNN versions PyTorch was built with; reported even when no GPU is visible
    info["torch_version"] = str(torch.__version__)
    info["cuda_version"] = torch.version.cuda
    with contextlib.suppress(RuntimeError):
        info["cudnn_version"] = torch.backends.cudnn.version()
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(device)
    info.update(
        gpu_name=props.name,
        gpu_total_memory_gb=round(props.total_memory / _GB, 2),
        gpu_compute_capability=f"{props.major}.{props.minor}",
        gpu_count=torch.cuda.device_count(),
    )
    return info


def reset_peak_memory(device: int = 0) -> None:
    """Reset the peak-memory counter (call at the start of a run). No-op without a GPU."""
    if cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: int = 0) -> float | None:
    """Return the peak GPU memory allocated by PyTorch in this process, in GB.

    This is what PyTorch allocated for tensors, a little below what ``nvidia-smi`` shows.
    Returns ``None`` without a GPU.
    """
    if not cuda_available():
        return None
    import torch

    return round(torch.cuda.max_memory_allocated(device) / _GB, 3)


def describe(info: dict[str, Any] | None = None) -> str:
    """One readable line for terminal output, e.g. ``NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB``.

    Pass the dict from :func:`gpu_info` to avoid querying the GPU again.
    """
    info = gpu_info() if info is None else info
    if info["gpu_name"] == "cpu":
        return "no CUDA GPU (CPU only)"
    name, memory = info["gpu_name"], info["gpu_total_memory_gb"]
    return f"{name}, {memory} GB, compute {info['gpu_compute_capability']}"
