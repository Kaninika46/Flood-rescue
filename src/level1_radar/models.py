"""U-Net models for the radar flood map and weight transfer between channel layouts.

``build_model(model_cfg, channels)`` returns a segmentation-models-pytorch network with one
output logit (flood). Initialisation (``model.init``):

* ``imagenet``: encoder weights from ImageNet. The first convolution is rebuilt for our channels:
  VV, VH and a third channel reuse the R, G, B filters; extra channels get the mean of the RGB
  filters.
* ``etci``: a released ETCI 2021 checkpoint (``model.etci_weights``, relative to ``$DATA_ROOT``).
  ETCI inputs are (VV, VH, 1 - VH/VV); our VV and VH reuse filters 0 and 1, other channels get
  the mean of the three. ETCI's 2-class softmax head becomes one logit exactly:
  ``sigmoid(z1 - z0) == softmax(z)[1]``.
* ``checkpoint``: one of our own checkpoints (``model.checkpoint``), adapted the same way.
* ``none``: random weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

from src.common.config import data_path

ARCHS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}
ETCI_CHANNELS = ("VV", "VH", "RATIO")
IMAGENET_CHANNELS = ("VV", "VH", "HAND")  # channel i reuses ImageNet filter i


def make_network(
    arch: str, encoder: str, in_channels: int, classes: int = 1, encoder_weights: str | None = None
) -> nn.Module:
    """Bare smp network."""
    try:
        cls = ARCHS[arch.lower()]
    except KeyError as err:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(ARCHS)}") from err
    return cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def first_conv_name(model: nn.Module) -> str:
    """Dotted name of the first ``Conv2d`` in the encoder."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name
    raise ValueError("model has no Conv2d")


def head_conv_name(model: nn.Module) -> str:
    """Dotted name of the last ``Conv2d`` (the segmentation head)."""
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = name
    if last is None:
        raise ValueError("model has no Conv2d")
    return last


def adapt_first_conv_weight(
    weight: torch.Tensor, src_channels: Sequence[str], dst_channels: Sequence[str]
) -> torch.Tensor:
    """Rebuild a first-layer weight ``(out, len(src), k, k)`` for ``dst_channels``.

    A channel present in both keeps its filter; a new channel gets the mean of all source
    filters. The result is rescaled so the summed input weight stays comparable.
    """
    if weight.shape[1] != len(src_channels):
        raise ValueError(f"weight has {weight.shape[1]} inputs, names list {len(src_channels)}")
    mean = weight.mean(dim=1)
    columns = []
    for name in dst_channels:
        if name in src_channels:
            columns.append(weight[:, list(src_channels).index(name)])
        else:
            columns.append(mean)
    new = torch.stack(columns, dim=1)
    return new * (len(src_channels) / len(dst_channels))


def binary_head_from_softmax(weight: torch.Tensor, bias: torch.Tensor | None) -> tuple:
    """Turn a 2-class softmax head into an equivalent single-logit head (class 1 vs 0)."""
    new_w = (weight[1] - weight[0]).unsqueeze(0)
    new_b = None if bias is None else (bias[1] - bias[0]).unsqueeze(0)
    return new_w, new_b


def transfer_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    src_channels: Sequence[str],
    dst_channels: Sequence[str],
) -> None:
    """Load ``state`` into ``model`` adapting the first convolution and a 2-class head."""
    state = dict(state)
    first = first_conv_name(model) + ".weight"
    state[first] = adapt_first_conv_weight(state[first], src_channels, dst_channels)
    head = head_conv_name(model)
    head_w = state[f"{head}.weight"]
    if head_w.shape[0] == 2 and model.get_submodule(head).out_channels == 1:
        w, b = binary_head_from_softmax(head_w, state.get(f"{head}.bias"))
        state[f"{head}.weight"] = w
        if b is not None:
            state[f"{head}.bias"] = b
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight transfer mismatch: missing {missing}, unexpected {unexpected}")


def build_model(model_cfg: Mapping[str, Any], channels: Sequence[str]) -> nn.Module:
    """Build the configured network for ``channels`` with the configured initialisation."""
    arch = model_cfg.get("arch", "unet")
    encoder = model_cfg.get("encoder", "resnet34")
    init = model_cfg.get("init", "imagenet")
    model = make_network(arch, encoder, in_channels=len(channels))
    if init == "none":
        return model
    if init == "imagenet":
        donor = make_network(arch, encoder, in_channels=3, encoder_weights="imagenet")
        src_channels = IMAGENET_CHANNELS
        state = donor.state_dict()  # its 1-logit head is freshly initialised, like ours
    elif init == "etci":
        state = torch.load(
            data_path(model_cfg["etci_weights"]), map_location="cpu", weights_only=True
        )
        src_channels = ETCI_CHANNELS
    elif init == "checkpoint":
        ckpt = torch.load(
            data_path(model_cfg["checkpoint"]), map_location="cpu", weights_only=False
        )
        state, src_channels = ckpt["model"], ckpt["channels"]
    else:
        raise ValueError(f"unknown model.init {init!r}")
    transfer_state(model, state, src_channels, channels)
    return model
