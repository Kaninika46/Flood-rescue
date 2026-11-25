"""Tests for src/common/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import (
    ConfigError,
    apply_overrides,
    data_path,
    effective_batch_size,
    get,
    get_data_root,
    load_config,
    run_name,
    save_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_sets_experiment_from_file_name(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "E4.yaml", "train:\n  batch_size: 8\n  accum_steps: 2\n"))
    assert cfg["experiment"] == "E4"
    assert effective_batch_size(cfg) == 16
    assert run_name(cfg["experiment"], 1) == "E4-s1"


def test_base_chain_merges_nested_sections(tmp_path: Path) -> None:
    _write(tmp_path / "E4.yaml", "train:\n  lr: 0.001\n  batch_size: 8\nchannels: [VV, VH]\n")
    child_text = "base: E4.yaml\nchannels: [VV, VH, HAND]\ntrain:\n  lr: 0.0005\n"
    child = _write(tmp_path / "E6.yaml", child_text)
    cfg = load_config(child)
    assert cfg["experiment"] == "E6"
    assert cfg["channels"] == ["VV", "VH", "HAND"]
    assert cfg["train"] == {"lr": 0.0005, "batch_size": 8}
    assert "base" not in cfg


def test_circular_base_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "base: b.yaml\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_overrides_parse_yaml_types() -> None:
    cfg = {"train": {"batch_size": 8, "amp": True, "encoder": "resnet34"}}
    out = apply_overrides(cfg, ["train.batch_size=4", "train.amp=false", "train.encoder=resnet50"])
    assert out["train"] == {"batch_size": 4, "amp": False, "encoder": "resnet50"}
    assert cfg["train"]["batch_size"] == 8  # original untouched


def test_override_typo_is_refused_unless_plus(tmp_path: Path) -> None:
    cfg = {"train": {"batch_size": 8}}
    with pytest.raises(ConfigError, match="not in the config"):
        apply_overrides(cfg, ["train.batchsize=4"])
    out = apply_overrides(cfg, ["+train.note=test", "+new.section.value=[1, 2]"])
    assert out["train"]["note"] == "test"
    assert out["new"]["section"]["value"] == [1, 2]
    with pytest.raises(ConfigError, match=r"key\.sub=value"):
        apply_overrides(cfg, ["train.batch_size"])


def test_get_with_dotted_keys() -> None:
    cfg = {"train": {"batch_size": 8}}
    assert get(cfg, "train.batch_size") == 8
    assert get(cfg, "train.missing", None) is None
    with pytest.raises(KeyError):
        get(cfg, "train.missing")
    assert effective_batch_size({"experiment": "E1"}) is None


def test_data_root_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path
    assert data_path("sen1floods11", "chip.tif") == tmp_path / "sen1floods11" / "chip.tif"
    with pytest.raises(ConfigError, match="relative"):
        data_path(str(tmp_path / "abs.tif"))


def test_missing_or_bad_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        get_data_root()
    monkeypatch.setenv("DATA_ROOT", "relative/path")
    with pytest.raises(ConfigError, match="absolute"):
        get_data_root()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "nope"))
    with pytest.raises(ConfigError, match="does not exist"):
        get_data_root()
    assert get_data_root(must_exist=False) == tmp_path / "nope"


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    cfg = {"experiment": "E4", "train": {"batch_size": 8, "lr": 0.001}}
    path = save_config(cfg, tmp_path / "out" / "config.yaml")
    assert load_config(path) == cfg
