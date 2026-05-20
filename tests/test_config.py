"""Tests for transcribe.config module."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from transcribe.config import load_config, resolve_device
from transcribe.data.types import PipelineConfig


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu() -> None:
    """Explicit 'cpu' should pass through unchanged."""
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda() -> None:
    """Explicit 'cuda' should pass through unchanged (no GPU check)."""
    assert resolve_device("cuda") == "cuda"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no YAML and no overrides, return a PipelineConfig with defaults."""
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        tmp_path / "nonexistent.yaml",
    )
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.device == "auto"
    assert cfg.diarize is True
    assert cfg.hotwords is None
    assert cfg.language == "zh"
    assert cfg.cache_dir == ".cache"
    assert cfg.num_speakers is None


def test_load_config_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML file values should override code defaults."""
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
            device: cpu
            language: en
            cache_dir: /tmp/my_cache
            num_speakers: 3
        """),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        tmp_path / "nonexistent.yaml",
    )
    cfg = load_config(config_path=str(yaml_file))
    assert cfg.device == "cpu"
    assert cfg.language == "en"
    assert cfg.cache_dir == "/tmp/my_cache"
    assert cfg.num_speakers == 3


def test_load_config_yaml_and_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI overrides should take precedence over YAML values."""
    yaml_file = tmp_path / "base.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
            device: cpu
            language: en
        """),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        tmp_path / "nonexistent.yaml",
    )
    cfg = load_config(
        config_path=str(yaml_file),
        cli_overrides={"device": "cuda", "language": "zh"},
    )
    assert cfg.device == "cuda"  # CLI wins
    assert cfg.language == "zh"  # CLI wins


def test_load_config_hotwords_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hotwords path supplied via CLI override."""
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        tmp_path / "nonexistent.yaml",
    )
    cfg = load_config(cli_overrides={"hotwords": "/path/to/hotwords.txt"})
    assert cfg.hotwords == "/path/to/hotwords.txt"


def test_load_config_ignores_none_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None-valued CLI overrides should not overwrite defaults or YAML."""
    yaml_file = tmp_path / "has_num_speakers.yaml"
    yaml_file.write_text("num_speakers: 5\n", encoding="utf-8")
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        tmp_path / "nonexistent.yaml",
    )
    cfg = load_config(
        config_path=str(yaml_file),
        cli_overrides={"num_speakers": None},
    )
    assert cfg.num_speakers == 5  # YAML preserved; None override ignored


def test_load_config_default_yaml_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no config_path given, _DEFAULT_CONFIG_PATH is consulted."""
    default_yaml = tmp_path / "config.yaml"
    default_yaml.write_text(
        textwrap.dedent("""\
            device: cpu
        """),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "transcribe.config._DEFAULT_CONFIG_PATH",
        default_yaml,
    )
    cfg = load_config()
    assert cfg.device == "cpu"
