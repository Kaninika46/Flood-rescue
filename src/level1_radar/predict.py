"""Flood probability and confidence from an ensemble of U-Nets, for chips or whole scenes (A8).

Each member is a checkpoint written by ``src/level1_radar/train.py`` (it stores its channels and
normalisation). A scene of any size is cut into overlapping ``tile x tile`` windows; window
predictions are blended with weights that fall off near tile edges, so seams disappear. Large
scenes are processed in blocks, so memory stays small.

Outputs two GeoTIFFs on the scene's grid:

* ``<stem>_flood_prob.tif``: mean water probability over members (0 to 1, NaN = no radar data)
* ``<stem>_flood_std.tif``: standard deviation over members, the **confidence map**
  (high = members disagree = low confidence; 0 for a single member)

    python -m src.level1_radar.predict --config configs/E6.yaml \\
        --s1 $DATA_ROOT/nepal/s1/S1_20240927.tif \\
        --hand $DATA_ROOT/nepal/terrain/S1_20240927_HAND.tif \\
        --out-dir $DATA_ROOT/nepal/level1
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import typer
from rasterio.windows import Window
from torch import nn

from src.common.checkpoint import load_checkpoint
from src.common.config import data_path, load_config
from src.common.logging import get_logger
from src.level1_radar.datasets import S1_CHANNELS, Norm, load_image
from src.level1_radar.models import make_network

log = get_logger("predict")


@dataclass
class Member:
    """One trained network with the channels and normalisation it was trained on."""

    model: nn.Module
    channels: list[str]
    norm: Norm
    name: str


def load_member(path: str | Path, device: torch.device) -> Member:
    """Load a ``best.pt`` / ``last.pt`` checkpoint written by the training script."""
    ckpt = load_checkpoint(path)
    model_cfg = ckpt["model_cfg"]
    model = make_network(
        model_cfg.get("arch", "unet"),
        model_cfg.get("encoder", "resnet34"),
        in_channels=len(ckpt["channels"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    norm = Norm(mean=ckpt["norm"]["mean"], std=ckpt["norm"]["std"])
    return Member(model, list(ckpt["channels"]), norm, Path(path).parent.name)


def window_weights(tile: int, overlap: int) -> np.ndarray:
    """Blending weights: 1 in the centre, a linear ramp over ``overlap`` pixels at the edges."""
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    ramp = np.clip(ramp / overlap, 1.0 / overlap, 1.0)
    return np.outer(ramp, ramp)


def tile_starts(length: int, tile: int, step: int) -> list[int]:
    """Window start positions covering ``length`` (the last window ends exactly at the edge)."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def _d4_inverse(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


class Ensemble:
    """Average of several members; also reports their standard deviation."""

    def __init__(
        self,
        members: Sequence[Member],
        device: torch.device | None = None,
        tile: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        amp: bool = True,
        tta: bool = False,
    ) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile, self.overlap, self.batch_size = tile, overlap, batch_size
        self.amp = amp and self.device.type == "cuda"
        self.tta = tta
        self.weights = window_weights(tile, overlap)

    @classmethod
    def from_checkpoints(cls, paths: Sequence[str | Path], **kwargs: Any) -> Ensemble:
        """Build from checkpoint files."""
        device = kwargs.pop("device", None) or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        members = [load_member(p, device) for p in paths]
        return cls(members, device=device, **kwargs)

    @property
    def channels(self) -> list[str]:
        """Every channel any member needs, in a stable order."""
        seen: list[str] = []
        for m in self.members:
            seen += [c for c in m.channels if c not in seen]
        return seen

    def load_layers(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """All needed channels of a split row, in physical units."""
        image = load_image(row, self.channels)
        return dict(zip(self.channels, image, strict=True))

    @torch.no_grad()
    def _forward(self, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        variants = [(k, f) for k in range(4) for f in (False, True)] if self.tta else [(0, False)]
        total = torch.zeros(batch.shape[0], batch.shape[-2], batch.shape[-1], device=self.device)
        for k, flip in variants:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = model(_d4(batch, k, flip))
            total += _d4_inverse(torch.sigmoid(logits.float()), k, flip)[:, 0]
        return total / len(variants)

    def _predict_member(self, member: Member, image: np.ndarray) -> np.ndarray:
        """Sliding-window probability for one member on a normalised ``(C, H, W)`` image."""
        _, height, width = image.shape
        pad_h, pad_w = max(0, self.tile - height), max(0, self.tile - width)
        if pad_h or pad_w:
            mode = "reflect" if pad_h < height and pad_w < width else "edge"
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
        _, ph, pw = image.shape
        step = self.tile - self.overlap
        boxes = [
            (r, c)
            for r in tile_starts(ph, self.tile, step)
            for c in tile_starts(pw, self.tile, step)
        ]
        acc = np.zeros((ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        for i in range(0, len(boxes), self.batch_size):
            chunk = boxes[i : i + self.batch_size]
            batch = np.stack([image[:, r : r + self.tile, c : c + self.tile] for r, c in chunk])
            probs = self._forward(member.model, torch.from_numpy(batch).to(self.device))
            for (r, c), prob in zip(chunk, probs.cpu().numpy(), strict=True):
                acc[r : r + self.tile, c : c + self.tile] += prob * self.weights
                wsum[r : r + self.tile, c : c + self.tile] += self.weights
        return (acc / wsum)[:height, :width]

    def predict(self, layers: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the members' water probability, ``(H, W)`` each."""
        probs = []
        for member in self.members:
            image = np.stack([layers[c] for c in member.channels]).astype(np.float32)
            probs.append(self._predict_member(member, member.norm.apply(image, member.channels)))
        stack = np.stack(probs)
        return stack.mean(axis=0), stack.std(axis=0)


def member_paths(cfg: Mapping[str, Any]) -> list[Path]:
    """Checkpoints of an experiment's ensemble: ``ensemble.members`` run names, else every seed."""
    names = cfg.get("ensemble", {}).get("members") or [
        f"{cfg['experiment']}-s{s}" for s in cfg.get("seeds", [0])
    ]
    return [data_path("checkpoints", n, "best.pt") for n in names]


def predict_scene(
    ensemble: Ensemble,
    s1_path: Path,
    out_dir: Path,
    terrain: Mapping[str, Path] | None = None,
    block: int = 4096,
) -> tuple[Path, Path]:
    """Predict a whole scene block by block and write the probability and std GeoTIFFs."""
    terrain = dict(terrain or {})
    missing = [c for c in ensemble.channels if c not in S1_CHANNELS and c not in terrain]
    if missing:
        raise ValueError(f"members need {missing}: pass --hand / --slope rasters")
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_path = out_dir / f"{s1_path.stem}_flood_prob.tif"
    std_path = out_dir / f"{s1_path.stem}_flood_std.tif"
    halo = ensemble.overlap
    with rasterio.open(s1_path) as s1:
        sources = {name: rasterio.open(path) for name, path in terrain.items()}
        for name, src in sources.items():
            if (src.crs, src.transform, src.shape) != (s1.crs, s1.transform, s1.shape):
                raise ValueError(
                    f"{name} raster is not on the S1 grid; build it with "
                    "scripts/prepare_hand.py raster <s1 scene>"
                )
        profile = {
            "driver": "GTiff",
            "height": s1.height,
            "width": s1.width,
            "count": 1,
            "dtype": "float32",
            "crs": s1.crs,
            "transform": s1.transform,
            "nodata": float("nan"),
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        with (
            rasterio.open(prob_path, "w", **profile) as dst_p,
            rasterio.open(std_path, "w", **profile) as dst_s,
        ):
            dst_p.set_band_description(1, "flood_probability")
            dst_s.set_band_description(1, "ensemble_std (high = low confidence)")
            for row in range(0, s1.height, block):
                for col in range(0, s1.width, block):
                    inner = Window(
                        col, row, min(block, s1.width - col), min(block, s1.height - row)
                    )
                    r0, c0 = max(0, row - halo), max(0, col - halo)
                    r1 = min(s1.height, row + inner.height + halo)
                    c1 = min(s1.width, col + inner.width + halo)
                    outer = Window(c0, r0, c1 - c0, r1 - r0)
                    vv_vh = s1.read([1, 2], window=outer).astype(np.float32)
                    layers = {"VV": vv_vh[0], "VH": vv_vh[1]}
                    for name, src in sources.items():
                        layers[name] = src.read(1, window=outer).astype(np.float32)
                    mean, std = ensemble.predict(layers)
                    no_data = np.isnan(vv_vh).all(axis=0)
                    mean[no_data], std[no_data] = np.nan, np.nan
                    sl = (
                        slice(row - r0, row - r0 + inner.height),
                        slice(col - c0, col - c0 + inner.width),
                    )
                    dst_p.write(mean[sl][None], window=inner)
                    dst_s.write(std[sl][None], window=inner)
                    log.info("block row %d col %d done", row, col)
        for src in sources.values():
            src.close()
    return prob_path, std_path


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    s1: Path = typer.Option(..., help="Sentinel-1 GeoTIFF: band 1 VV, band 2 VH, in dB."),
    out_dir: Path = typer.Option(..., help="Folder for the two output GeoTIFFs."),
    config: Path = typer.Option(
        None, help="Experiment config; members = its seeds or ensemble.members."
    ),
    checkpoint: list[Path] = typer.Option([], help="Explicit member checkpoints instead."),
    hand: Path = typer.Option(None, help="HAND raster on the S1 grid (needed for HAND models)."),
    slope: Path = typer.Option(None, help="Slope raster on the S1 grid."),
    tta: bool = typer.Option(False, help="Average the 8 flips/rotations (slower)."),
    tile: int = typer.Option(512),
    overlap: int = typer.Option(64),
) -> None:
    """Write ``<stem>_flood_prob.tif`` and ``<stem>_flood_std.tif``."""
    paths = list(checkpoint) or member_paths(load_config(config))
    ensemble = Ensemble.from_checkpoints(paths, tile=tile, overlap=overlap, tta=tta)
    terrain = {k: v for k, v in {"HAND": hand, "SLOPE": slope}.items() if v is not None}
    prob, std = predict_scene(ensemble, s1, out_dir, terrain)
    log.info("wrote %s and %s", prob, std)


if __name__ == "__main__":
    app()
