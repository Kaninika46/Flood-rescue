"""SpaceNet 5 road weights -> initial weights for our foundation network (S2, prompt B5).

Cheapest option that fits a 6 GB GPU: the SpaceNet 5 baseline (CRESI) model, 98 MB,
``s3://spacenet-dataset/spacenet-model-weights/spacenet-5/baseline/weights/fold0_best.pth``.
It is a ResNet-34 U-Net of the same code family as the SpaceNet 8 baseline (``pytorch_zoo``), so
encoder and decoder tensors map one to one onto our foundation network; only the output layer
differs (SN5 predicts road speed only).

The file is a whole pickled model (old PyTorch format), not a state_dict. A static scan of the
pickle (``pickletools``, nothing executed) shows it references only PyTorch layers and
``pytorch_zoo.{unet,abstract_model,resnet}`` classes, which are our copies in
``src/level2_roads/sn8/models``. The script first tries ``torch.load(weights_only=True)`` with an
allowlist of exactly those classes; PyTorch's safe loader cannot read this old format, so the
full load needs ``--trust-pickle`` (your decision: it runs whatever the file contains). The
result is saved as a plain tensor state_dict, which training loads with ``weights_only=True``.

    python scripts/prepare_sn5_weights.py download
    python scripts/prepare_sn5_weights.py convert --trust-pickle   # -> weights/sn5/...converted.pth
"""

from __future__ import annotations

import collections
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import typer
from torch import nn

from src.common.config import data_path
from src.common.logging import get_logger

URL = (
    "https://spacenet-dataset.s3.amazonaws.com/spacenet-model-weights/spacenet-5/baseline/"
    "weights/fold0_best.pth"
)
RAW = "weights/sn5/fold0_best.pth"
CONVERTED = "weights/sn5/sn5_xdxd_resnet34_converted.pth"
log = get_logger("sn5")
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def download() -> None:
    """Fetch the 98 MB SpaceNet 5 baseline model."""
    path = data_path(RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
    log.info("downloaded %s (%.0f MB)", path, path.stat().st_size / 1e6)


def allowlist() -> list[tuple[object, str]]:
    """The only classes the pickle may build, under the names it uses."""
    from src.level2_roads.sn8.models import abstract_model, resnet, unet

    return [
        (collections.OrderedDict, "collections.OrderedDict"),
        (unet.Resnet34_upsample, "pytorch_zoo.unet.Resnet34_upsample"),
        (abstract_model.ConvBottleneck, "pytorch_zoo.abstract_model.ConvBottleneck"),
        (abstract_model.UnetDecoderBlock, "pytorch_zoo.abstract_model.UnetDecoderBlock"),
        (resnet.BasicBlock, "pytorch_zoo.resnet.BasicBlock"),
        (nn.ReLU, "torch.nn.modules.activation.ReLU"),
        (nn.BatchNorm2d, "torch.nn.modules.batchnorm.BatchNorm2d"),
        (nn.ModuleList, "torch.nn.modules.container.ModuleList"),
        (nn.Sequential, "torch.nn.modules.container.Sequential"),
        (nn.Conv2d, "torch.nn.modules.conv.Conv2d"),
        (nn.MaxPool2d, "torch.nn.modules.pooling.MaxPool2d"),
        (nn.Upsample, "torch.nn.modules.upsampling.Upsample"),
    ]


def load_sn5(trust_pickle: bool) -> object:
    """Safe load if PyTorch allows it; the full pickle load only with ``--trust-pickle``."""
    import pickle

    from src.level2_roads.sn8 import models
    from src.level2_roads.sn8.models import abstract_model, resnet, unet

    try:
        with torch.serialization.safe_globals(allowlist()):
            return torch.load(data_path(RAW), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as err:
        if not trust_pickle:
            raise typer.Exit(
                "PyTorch's safe loader cannot read this 2019-format file (it embeds module source "
                "code). A static scan found only PyTorch layers and pytorch_zoo classes in it. "
                "If you trust the official SpaceNet bucket, rerun with --trust-pickle, which "
                f"unpickles it in full. ({err.__class__.__name__})"
            ) from err
    # the pickle names our copied classes "pytorch_zoo.*"; point those names at our copies
    sys.modules.update(
        {
            "pytorch_zoo": models,
            "pytorch_zoo.unet": unet,
            "pytorch_zoo.abstract_model": abstract_model,
            "pytorch_zoo.resnet": resnet,
        }
    )
    return torch.load(data_path(RAW), map_location="cpu", weights_only=False)


@app.command()
def convert(
    trust_pickle: bool = typer.Option(False, help="Allow the full (code-executing) pickle load."),
) -> None:
    """Load the SN5 model, map tensors onto our foundation network, save a plain state_dict."""
    from src.level2_roads.sn8.models import abstract_model, unet

    obj = load_sn5(trust_pickle)
    state = obj.state_dict() if isinstance(obj, nn.Module) else dict(obj)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    abstract_model.AbstractModel.initialize_encoder = lambda self, model, url: None
    ours = unet.Resnet34_upsample(num_classes=[1, 8], num_channels=3).state_dict()
    matched = {k: v for k, v in state.items() if k in ours and ours[k].shape == v.shape}
    missing = [k for k in ours if k not in matched]
    out = data_path(CONVERTED)
    torch.save(matched, out)
    log.info(
        "SN5 tensors %d, ours %d, transferred %d; not transferred: %s",
        len(state),
        len(ours),
        len(matched),
        missing[:10],
    )
    log.info("wrote %s (configs/S2.yaml model.init_weights.foundation)", out)


if __name__ == "__main__":
    app()
