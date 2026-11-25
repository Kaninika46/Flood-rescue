"""Check the cloud/haze mask by eye on 10 post-event tiles (prompt B6).

    python scripts/cloud_check.py --split sn8_val            # or nepal_maxar_labelled
    python scripts/cloud_check.py --split sn8_val --set cloud.brightness_min=160

Writes ``docs/figures/cloud_check/<tile>.png`` (post image | mask in red) and prints the cloud
share per tile. Tune the thresholds in ``configs/S3.yaml`` until clouds and haze are covered and
bright roofs and roads are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import typer

from src.common.config import data_path, load_config, repo_root
from src.level2_roads.cloud import cloud_mask
from src.level2_roads.data import read_image, read_sn8_split, warp_post_to_pre

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    split: str = typer.Option("sn8_val"),
    n: int = typer.Option(10),
    config: Path = typer.Option(Path("configs/S3.yaml")),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Render ``n`` tiles with their cloud masks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(repo_root() / config, set_)
    rows = read_sn8_split(split)
    rows = rows[[data_path(p).exists() for p in rows["postimg"]]].head(n)
    out = repo_root() / "docs" / "figures" / "cloud_check"
    out.mkdir(parents=True, exist_ok=True)
    for row in rows.to_dict("records"):
        _, h, w = read_image(data_path(row["preimg"])).shape
        post = warp_post_to_pre(data_path(row["postimg"]), h, w)
        mask = cloud_mask(post, cfg.get("cloud"))
        rgb = np.moveaxis(post[:3], 0, -1).astype(np.uint8)
        overlay = rgb.copy()
        overlay[mask] = (0.5 * overlay[mask] + [127, 0, 0]).astype(np.uint8)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rgb)
        ax[1].imshow(overlay)
        for a, t in zip(ax, ["post-event", f"cloud/haze {mask.mean():.1%}"], strict=True):
            a.set_title(f"{row['tile_id']}: {t}", fontsize=9)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{row['tile_id']}.png", dpi=70)
        plt.close(fig)
        print(f"{row['tile_id']}: cloud/haze {mask.mean():.1%}")


if __name__ == "__main__":
    app()
