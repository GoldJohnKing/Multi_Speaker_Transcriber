"""Configuration loading for the transcription pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from transcribe.data.types import PipelineConfig

# Known PipelineConfig field names
_PIPELINE_CONFIG_FIELDS = frozenset(
    {"device", "denoise", "diarize", "separate", "tse", "hotwords", "language", "cache_dir", "num_speakers"}
)

# Default config file location: project root / config.yaml
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to 'cuda' or 'cpu'.

    Parameters
    ----------
    device : str
        One of 'auto', 'cuda', or 'cpu'.

    Returns
    -------
    str
        'cuda' if available and device is 'auto', otherwise the input value.
    """
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> PipelineConfig:
    """Load pipeline configuration with priority: CLI > YAML > code defaults.

    Parameters
    ----------
    config_path : str | None
        Explicit path to a YAML config file. If *None*, falls back to
        ``config.yaml`` in the project root (if it exists).
    cli_overrides : dict | None
        Keyword overrides from the CLI. ``None`` values are ignored.

    Returns
    -------
    PipelineConfig
        Merged configuration.
    """
    # 1. Code defaults (empty – PipelineConfig dataclass supplies defaults)
    merged: dict = {}

    # 2. YAML config
    yaml_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as fh:
            yaml_cfg = yaml.safe_load(fh) or {}
        # Merge top-level keys only
        for key, value in yaml_cfg.items():
            if key in _PIPELINE_CONFIG_FIELDS:
                merged[key] = value

    # 3. CLI overrides (skip None values)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if key in _PIPELINE_CONFIG_FIELDS and value is not None:
                merged[key] = value

    return PipelineConfig(**merged)
