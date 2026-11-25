"""Experiment configuration: YAML files, ``DATA_ROOT`` and command-line overrides.

Conventions (see README "Configs"):

* One YAML per experiment ID in ``configs/`` (``E4.yaml``, ``S0.yaml``...). The ``experiment``
  key defaults to the file name without ``.yaml``.
* A config may start from another with ``base: E4.yaml`` (path relative to the config file);
  the child's keys win, nested dicts are merged.
* Training settings live under ``train:``; ``train.batch_size x train.accum_steps`` is the
  effective batch size and must stay the same for every run of an experiment.
* Data paths in configs are relative to ``$DATA_ROOT``; resolve them with :func:`data_path`.
* Overrides from the command line use ``key.sub=value`` (value parsed as YAML). Overriding a
  key that does not exist is an error (catches typos); prefix with ``+`` to add a new key.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

DATA_ROOT_ENV = "DATA_ROOT"
_MISSING = object()


class ConfigError(ValueError):
    """Raised for a bad config file, a bad override or a missing ``DATA_ROOT``."""


def repo_root() -> Path:
    """Return the repository root (the folder that holds ``README.md`` and ``src/``)."""
    return Path(__file__).resolve().parents[2]


def get_data_root(must_exist: bool = True) -> Path:
    """Return ``$DATA_ROOT`` as an absolute path.

    Raises:
        ConfigError: if the variable is unset, not absolute, or (with ``must_exist``)
            not an existing directory.
    """
    value = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not value:
        raise ConfigError(
            f"{DATA_ROOT_ENV} is not set. Add a line like "
            f"`export {DATA_ROOT_ENV}=$HOME/flood-data` to ~/.bashrc (inside WSL on Windows), "
            "open a new terminal and try again."
        )
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ConfigError(f"{DATA_ROOT_ENV} must be an absolute path, got {value!r}")
    if must_exist and not root.is_dir():
        raise ConfigError(f"{DATA_ROOT_ENV}={root} does not exist or is not a directory")
    return root


def data_path(*parts: str | os.PathLike[str], must_exist: bool = False) -> Path:
    """Join path parts onto ``$DATA_ROOT``.

    Parts must be relative: absolute paths are refused so no machine-specific path can
    sneak into a config or a split file.
    """
    relative = Path(*parts)
    if relative.is_absolute():
        raise ConfigError(f"data paths must be relative to ${DATA_ROOT_ENV}, got {relative}")
    path = get_data_root(must_exist=False) / relative
    if must_exist and not path.exists():
        raise ConfigError(f"missing data file or folder: {path}")
    return path


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping (an empty file gives ``{}``)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` updated by ``override``, merging nested dicts recursively."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_base(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ConfigError(f"circular 'base' chain: {chain}")
    cfg = load_yaml(path)
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    base_cfg = _load_with_base(path.parent / str(base), (*seen, path))
    base_cfg.pop("experiment", None)  # a child never inherits its parent's experiment ID
    return deep_merge(base_cfg, cfg)


def parse_override(text: str) -> tuple[str, Any, bool]:
    """Parse ``key.sub=value`` (or ``+key.sub=value``) into ``(dotted_key, value, is_new)``."""
    if "=" not in text:
        raise ConfigError(f"override must look like key.sub=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    is_new = key.startswith("+")
    key = key.removeprefix("+")
    if not key or any(not part for part in key.split(".")):
        raise ConfigError(f"bad override key in {text!r}")
    try:
        value = yaml.safe_load(raw) if raw.strip() else None
    except yaml.YAMLError as err:
        raise ConfigError(f"cannot parse value in override {text!r}: {err}") from err
    return key, value, is_new


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with ``key.sub=value`` overrides applied in order."""
    out = copy.deepcopy(dict(cfg))
    for text in overrides:
        key, value, is_new = parse_override(text)
        *parents, leaf = key.split(".")
        node = out
        for part in parents:
            child = node.get(part, _MISSING)
            if child is _MISSING and is_new:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ConfigError(f"override {text!r}: '{part}' is not a section in the config")
            node = child
        if leaf not in node and not is_new:
            raise ConfigError(
                f"override {text!r}: key '{key}' is not in the config "
                f"(check the spelling, or write +{key}=... to add a new key)"
            )
        node[leaf] = value
    return out


def load_config(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load an experiment config, resolve its ``base`` chain and apply overrides.

    Args:
        path: The YAML file, e.g. ``configs/E4.yaml``.
        overrides: Strings like ``train.batch_size=4`` from the command line.

    Returns:
        A plain nested dict that always has an ``experiment`` key.
    """
    path = Path(path)
    cfg = _load_with_base(path, ())
    cfg.setdefault("experiment", path.stem)
    return apply_overrides(cfg, overrides)


def get(cfg: Mapping[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Read ``cfg["a"]["b"]`` as ``get(cfg, "a.b")``.

    Raises ``KeyError`` for a missing key unless ``default`` is given.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted_key)
            return default
        node = node[part]
    return node


def run_name(experiment: str, seed: int) -> str:
    """Return the standard run name ``<experiment>-s<seed>``, e.g. ``E4-s1``."""
    return f"{experiment}-s{seed}"


def effective_batch_size(cfg: Mapping[str, Any]) -> int | None:
    """Return ``train.batch_size x train.accum_steps`` (``accum_steps`` defaults to 1).

    Returns ``None`` when the config has no ``train.batch_size`` (e.g. non-training runs).
    """
    batch = get(cfg, "train.batch_size", None)
    if batch is None:
        return None
    accum = get(cfg, "train.accum_steps", 1)
    return int(batch) * int(accum)


def to_plain(value: Any) -> Any:
    """Recursively turn str/int/float subclasses and NumPy scalars into plain Python values."""
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        return value.item()
    for kind in (str, int, float):
        if isinstance(value, kind):
            return kind(value)
    return str(value)


def save_config(cfg: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a resolved config to YAML (keeps key order) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False, allow_unicode=True)
    return path
