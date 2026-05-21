"""Tests for the Qwen3-ASR backend."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from transcribe.models.asr import list_backends
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import _BACKENDS


def test_qwen3_asr_registered() -> None:
    """"Qwen3-ASR" should appear in the backend registry."""
    assert "Qwen3-ASR" in list_backends()


def test_qwen3_asr_is_asr_base_subclass() -> None:
    """The registered class should be an ASRBase subclass."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber
    assert issubclass(Qwen3ASRTranscriber, ASRBase)


def test_supports_hotwords_true() -> None:
    """Qwen3-ASR should declare hotword support (via context)."""
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    assert instance.supports_hotwords is True


def test_import_error_without_qwen_asr() -> None:
    """Instantiation without qwen-asr installed should give a helpful error."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    with patch.dict(sys.modules, {"qwen_asr": None}):
        with pytest.raises(ImportError, match="uv sync --extra qwen-asr"):
            Qwen3ASRTranscriber(device="cpu")


def test_load_context_reads_hotword_file(tmp_path) -> None:
    """_load_context should read hotword file and join with spaces."""
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)

    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n王五\n", encoding="utf-8")

    result = instance._load_context(str(hw_file))
    assert result == "张三 李四 王五"


def test_load_context_none_returns_empty() -> None:
    """_load_context with None path returns empty string."""
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    assert instance._load_context(None) == ""


def test_load_context_empty_file_returns_empty() -> None:
    """_load_context with empty file returns empty string."""
    import tempfile
    import pathlib

    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name
    result = instance._load_context(path)
    assert result == ""
    pathlib.Path(path).unlink()
