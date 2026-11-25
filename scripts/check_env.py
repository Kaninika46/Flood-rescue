"""Check this laptop's "flood" environment.

Prints Python, PyTorch, CUDA, GPU name and memory, the versions of the key packages and the
state of ``$DATA_ROOT``, then runs one mixed-precision training step (forward and backward) of
a small U-Net. Exit code 0 means everything is ready; 1 means something needs fixing.

    python scripts/check_env.py                                  # on a training laptop
    python scripts/check_env.py --cpu-ok                         # machine without a GPU
    python scripts/check_env.py | tee docs/laptops/<name>.txt    # prompt P3
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common import gpu
from src.common.config import DATA_ROOT_ENV
from src.common.logging import git_info

# (import name, label) for every package the "flood" env must provide
PACKAGES: list[tuple[str, str]] = [
    ("torch", "pytorch"),
    ("torchvision", "torchvision"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("albumentations", "albumentations"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("lightgbm", "lightgbm"),
    ("osgeo.gdal", "gdal"),
    ("rasterio", "rasterio"),
    ("geopandas", "geopandas"),
    ("pyproj", "pyproj"),
    ("shapely", "shapely"),
    ("osmnx", "osmnx"),
    ("networkx", "networkx"),
    ("pystac", "pystac"),
    ("boto3", "boto3"),
    ("ee", "earthengine-api"),
    ("wandb", "wandb"),
    ("mlflow", "mlflow"),
    ("typer", "typer"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
]

app = typer.Typer(add_completion=False, help=__doc__)


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def check_packages() -> list[str]:
    """Import every package and print its installed version; return the names that failed."""
    missing = []
    for module, label in PACKAGES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # deprecation chatter from dependencies
                mod = importlib.import_module(module)
            print(f"  {label:30s} {_version(label, mod)}")
        except Exception as err:  # any import error is a broken install
            print(f"  {label:30s} MISSING ({type(err).__name__}: {err})")
            missing.append(label)
    return missing


def _version(label: str, module: object) -> str:
    """Installed distribution version (more reliable than ``__version__``)."""
    try:
        return metadata.version("torch" if label == "pytorch" else label)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "?"))


def check_data_root() -> None:
    """Report ``$DATA_ROOT``: set, existing, free space, and not on a slow WSL Windows mount.

    Only warns: data is not needed until the first download prompt.
    """
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        print(f"  {DATA_ROOT_ENV} is not set (needed before any data step; see README)")
        return
    root = Path(value).expanduser()
    print(f"  {DATA_ROOT_ENV}={root}")
    if not root.is_dir():
        print('  folder does not exist yet: mkdir -p "$DATA_ROOT"')
        return
    free_gb = shutil.disk_usage(root).free / 1024**3
    print(f"  free space: {free_gb:.0f} GB")
    if free_gb < 60:
        print("  WARNING: under 60 GB free; Sen1Floods11, HAND and SpaceNet 8 need more")
    if _is_wsl() and str(root.resolve()).startswith("/mnt/"):
        print(
            "  WARNING: data on a Windows drive (/mnt/...) is very slow from WSL;"
            " keep it in the Linux home folder"
        )


def _is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def amp_training_step(device: str, batch_size: int, size: int) -> dict[str, float | str]:
    """One mixed-precision forward + backward + optimizer step of a ResNet-18 U-Net.

    Random weights (no download). fp16 with a GradScaler on CUDA, bfloat16 on CPU.
    """
    import segmentation_models_pytorch as smp
    import torch

    torch.manual_seed(0)
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    x = torch.randn(batch_size, 3, size, size, device=device)
    y = (torch.rand(batch_size, 1, size, size, device=device) > 0.5).float()

    gpu.reset_peak_memory()
    start = time.perf_counter()
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss is not finite: {loss.item()}")
    return {
        "output_shape": str(tuple(logits.shape)),
        "autocast_dtype": str(logits.dtype).replace("torch.", ""),
        "loss": round(loss.item(), 4),
        "seconds": round(seconds, 2),
        "peak_memory_gb": gpu.peak_memory_gb() or 0.0,
    }


@app.command()
def main(
    cpu_ok: bool = typer.Option(False, "--cpu-ok", help="Do not fail when there is no CUDA GPU."),
    batch_size: int = typer.Option(2, help="Batch size for the test step."),
    size: int = typer.Option(256, help="Crop size (pixels) for the test step."),
) -> None:
    """Print the environment report and run the mixed-precision test step."""
    problems: list[str] = []

    _section("System")
    print(f"  Python      {platform.python_version()} ({sys.executable})")
    print(f"  OS          {platform.platform(terse=True)}{'  [WSL2]' if _is_wsl() else ''}")
    print(f"  conda env   {os.environ.get('CONDA_DEFAULT_ENV', '(none active)')}")
    commit = git_info()
    print(f"  git commit  {commit['git_commit'][:10]} on {commit['git_branch']}")
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python {platform.python_version()} (project uses 3.11)")

    _section("Packages")
    missing = check_packages()
    problems += [f"missing package {name}" for name in missing]

    _section("PyTorch and GPU")
    info = gpu.gpu_info()
    print(f"  PyTorch     {info['torch_version']}")
    print(f"  CUDA build  {info['cuda_version'] or 'none'}   cuDNN {info['cudnn_version'] or '-'}")
    print(f"  GPU         {gpu.describe(info)}")
    device = "cuda" if gpu.cuda_available() else "cpu"
    if device == "cuda":
        import torch

        arch = "sm_{}{}".format(*torch.cuda.get_device_capability(0))
        supported = torch.cuda.get_arch_list()
        print(f"  GPU count   {info['gpu_count']}   arch {arch}")
        if supported and arch not in supported:
            print(
                f"  WARNING: PyTorch was built for {supported}, not {arch};"
                " the test step below shows whether it still works"
            )
    elif not cpu_ok:
        problems.append("no CUDA GPU visible to PyTorch (driver? WSL GPU? use --cpu-ok to skip)")

    _section(f"Mixed-precision U-Net step ({device}, batch {batch_size}, {size}x{size})")
    if "pytorch" in missing or "segmentation-models-pytorch" in missing:
        print("  skipped (PyTorch or segmentation-models-pytorch missing)")
    else:
        try:
            for key, value in amp_training_step(device, batch_size, size).items():
                print(f"  {key:16s} {value}")
        except Exception as err:
            print(f"  FAILED: {type(err).__name__}: {err}")
            problems.append(f"AMP training step failed: {err}")

    _section("Data")
    check_data_root()

    _section("Result")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        raise typer.Exit(code=1)
    print("  OK: this laptop is ready")


if __name__ == "__main__":
    app()
