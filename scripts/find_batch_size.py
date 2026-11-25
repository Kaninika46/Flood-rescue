"""Find the largest training batch that fits on this GPU (prompt P3).

Runs real mixed-precision training steps (forward, backward, optimizer) of a U-Net with random
weights and doubles the batch until the GPU runs out of memory, then bisects. The result keeps
a safety margin, because real training also holds the data loader, validation and fragmentation.

    python scripts/find_batch_size.py                       # ResNet-34 U-Net, 3 channels, 256 crops
    python scripts/find_batch_size.py --channels 2 --encoder mobilenet_v2
    python scripts/find_batch_size.py --effective 16        # also suggest batch_size x accum_steps
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common import gpu

app = typer.Typer(add_completion=False, help=__doc__)


def fits(model: Any, optimizer: Any, scaler: Any, batch: int, channels: int, crop: int) -> bool:
    """Try one AMP training step at this batch size; False on out-of-memory."""
    import torch

    try:
        x = torch.randn(batch, channels, crop, crop, device="cuda")
        y = (torch.rand(batch, 1, crop, crop, device="cuda") > 0.5).float()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return True
    except torch.OutOfMemoryError:
        return False
    finally:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


@app.command()
def main(
    encoder: str = typer.Option("resnet34", help="segmentation-models-pytorch encoder name."),
    channels: int = typer.Option(3, help="Input channels (VV, VH, HAND = 3)."),
    crop: int = typer.Option(256, help="Training crop size in pixels."),
    max_batch: int = typer.Option(128, help="Upper limit to search."),
    margin: float = typer.Option(0.8, help="Keep this fraction of the largest batch that fits."),
    effective: int = typer.Option(16, help="Effective batch size the configs use."),
) -> None:
    """Search the largest batch and print a recommended batch_size / accum_steps pair."""
    import segmentation_models_pytorch as smp
    import torch

    if not gpu.cuda_available():
        print("No CUDA GPU visible; run this on the training laptop.")
        raise typer.Exit(code=1)
    print(f"GPU: {gpu.describe()}")
    model = smp.Unet(encoder, encoder_weights=None, in_channels=channels, classes=1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    low, high = 0, 1
    while high <= max_batch and fits(model, optimizer, scaler, high, channels, crop):
        low, high = high, high * 2
    high = min(high, max_batch + 1)
    while high - low > 1:
        mid = (low + high) // 2
        if fits(model, optimizer, scaler, mid, channels, crop):
            low = mid
        else:
            high = mid
    if low == 0:
        print("Even batch 1 does not fit. Use a smaller crop or encoder.")
        raise typer.Exit(code=1)

    gpu.reset_peak_memory()
    fits(model, optimizer, scaler, low, channels, crop)
    safe = max(1, int(low * margin))
    # largest power of two <= safe that divides the effective batch size
    batch = 1
    while batch * 2 <= safe and effective % (batch * 2) == 0:
        batch *= 2
    print(f"Unet-{encoder}, {channels} channels, {crop}x{crop} crops, mixed precision:")
    print(f"  largest batch that fits: {low} (peak {gpu.peak_memory_gb()} GB)")
    print(f"  safe batch ({margin:.0%}):      {safe}")
    print(
        f"  recommended for effective batch {effective}: "
        f"train.batch_size={batch} train.accum_steps={effective // batch}"
    )


if __name__ == "__main__":
    app()
