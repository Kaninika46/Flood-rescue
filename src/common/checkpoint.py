"""Load our own checkpoints with ``torch.load(weights_only=True)``.

``weights_only=True`` unpickles only tensors, plain containers and an allowlist of types, so a
tampered ``.pt`` file cannot run code when it is loaded. Our files need:

* ``best.pt`` (Level 1 and Level 2): nothing beyond the default allowlist.
* ``last.pt`` (resume): the NumPy RNG state saved by ``level1_radar.train.rng_state`` is a NumPy
  array, so resuming also allows ``numpy.ndarray``, ``numpy.dtype`` (and its concrete dtype
  classes) and NumPy's array rebuild function. Nothing else.

Pretrained files from other projects are loaded where they are used (``etci_reproduce`` already
uses ``weights_only=True``; ``scripts/prepare_sn5_weights.py`` converts the SpaceNet 5 file once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def numpy_globals() -> list[Any]:
    """The NumPy types an RNG-state array needs (NumPy 1.x and 2.x)."""
    try:
        from numpy._core.multiarray import _reconstruct  # NumPy 2
    except ImportError:  # NumPy 1.x (env sn8)
        from numpy.core.multiarray import _reconstruct
    # the concrete dtype classes are pickled too: numpy.dtypes.UInt32DType in NumPy >= 1.25,
    # numpy.dtype[uint32] before; type(np.dtype(...)) is the right class in both
    dtype_classes = [type(np.dtype(t)) for t in ("u4", "i8", "f8")]
    return [_reconstruct, np.ndarray, np.dtype, *dtype_classes]


def load_checkpoint(path: str | Path, resume: bool = False) -> Any:
    """``torch.load`` on the CPU with ``weights_only=True``.

    ``resume=True`` is for ``last.pt`` files (optimizer, scheduler, scaler and RNG state); it adds
    only the NumPy array types from :func:`numpy_globals` to the allowlist, for this call only.
    """
    if not resume:
        return torch.load(path, map_location="cpu", weights_only=True)
    with torch.serialization.safe_globals(numpy_globals()):
        return torch.load(path, map_location="cpu", weights_only=True)
