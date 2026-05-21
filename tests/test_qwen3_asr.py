"""Tests for the Qwen3-ASR backend."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
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


def test_load_context_missing_file_returns_empty() -> None:
    """_load_context with missing file returns empty string (no crash)."""
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    result = instance._load_context("/nonexistent/path/hotwords.txt")
    assert result == ""


def test_transcribe_with_mocked_model() -> None:
    """transcribe() should preserve punctuation from text in output."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    TS = types.SimpleNamespace
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    instance._context = ""
    instance._language = None

    # Real model behaviour: text has punctuation, time_stamps do not
    mock_result = TS(
        text="你好。",
        time_stamps=[
            TS(text="你", start_time=0.0, end_time=0.2),
            TS(text="好", start_time=0.2, end_time=0.4),
        ],
    )
    instance._model = types.SimpleNamespace(
        transcribe=lambda **kw: [mock_result]
    )

    audio = AudioSegment(
        waveform=np.zeros(8000, dtype=np.float32),
        sample_rate=16000,
        start_time=1.0,
        end_time=2.0,
    )
    result = instance.transcribe(audio)
    assert len(result) == 1
    assert result[0].text == "你好"  # sentence-end 。 discarded
    assert result[0].start_time == pytest.approx(1.0)
    assert result[0].end_time == pytest.approx(1.42)  # 1.0 + 0.4 + 0.02 (punc interp)


def test_transcribe_empty_result() -> None:
    """transcribe() returns empty list when model returns no text."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    instance._context = ""
    instance._language = None
    instance._model = types.SimpleNamespace(
        transcribe=lambda **kw: [types.SimpleNamespace(text="", time_stamps=None)]
    )

    audio = AudioSegment(
        waveform=np.zeros(8000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)
    assert result == []


def test_transcribe_no_timestamps_fallback() -> None:
    """transcribe() falls back to audio boundaries when no timestamps."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    instance._context = ""
    instance._language = None
    instance._model = types.SimpleNamespace(
        transcribe=lambda **kw: [
            types.SimpleNamespace(text="你好", time_stamps=None)
        ]
    )

    audio = AudioSegment(
        waveform=np.zeros(8000, dtype=np.float32),
        sample_rate=16000,
        start_time=5.0,
        end_time=7.0,
    )
    result = instance.transcribe(audio)
    assert len(result) == 1
    assert result[0].text == "你好"
    assert result[0].start_time == pytest.approx(5.0)
    assert result[0].end_time == pytest.approx(7.0)


# --- _align_text_to_timestamps tests ---


def test_align_basic_without_punctuation() -> None:
    """Alignment with no punctuation is identity mapping."""
    from transcribe.models.asr.qwen3_asr import _align_text_to_timestamps

    TS = types.SimpleNamespace
    time_stamps = [
        TS(text="你", start_time=0.0, end_time=0.2),
        TS(text="好", start_time=0.2, end_time=0.4),
    ]
    result = _align_text_to_timestamps("你好", time_stamps, offset=0.0)
    assert len(result) == 2
    assert result[0] == ("你", 0.0, 0.2)
    assert result[1] == ("好", 0.2, 0.4)


def test_align_preserves_punctuation() -> None:
    """Punctuation chars get interpolated timestamps."""
    from transcribe.models.asr.qwen3_asr import _align_text_to_timestamps

    TS = types.SimpleNamespace
    time_stamps = [
        TS(text="你", start_time=0.0, end_time=0.2),
        TS(text="好", start_time=0.3, end_time=0.5),
    ]
    # "。" is in text but not in time_stamps
    result = _align_text_to_timestamps("你好。", time_stamps, offset=0.0)
    assert len(result) == 3
    assert result[0] == ("你", 0.0, 0.2)
    assert result[1] == ("好", 0.3, 0.5)
    assert result[2][0] == "。"
    assert result[2][1] == pytest.approx(0.5)  # prev_end
    assert result[2][2] == pytest.approx(0.52)  # prev_end + 0.02


def test_align_multi_char_token() -> None:
    """Multi-char tokens like 'NPC' are flattened with distributed time."""
    from transcribe.models.asr.qwen3_asr import _align_text_to_timestamps

    TS = types.SimpleNamespace
    time_stamps = [
        TS(text="NPC", start_time=1.0, end_time=1.3),
    ]
    result = _align_text_to_timestamps("NPC", time_stamps, offset=0.0)
    assert len(result) == 3
    assert result[0] == ("N", pytest.approx(1.0), pytest.approx(1.1))
    assert result[1] == ("P", pytest.approx(1.1), pytest.approx(1.2))
    assert result[2] == ("C", pytest.approx(1.2), pytest.approx(1.3))


def test_align_with_offset() -> None:
    """Offset is applied to all timestamps."""
    from transcribe.models.asr.qwen3_asr import _align_text_to_timestamps

    TS = types.SimpleNamespace
    time_stamps = [
        TS(text="你", start_time=0.0, end_time=0.2),
    ]
    result = _align_text_to_timestamps("你", time_stamps, offset=10.0)
    assert result[0] == ("你", pytest.approx(10.0), pytest.approx(10.2))


def test_align_punctuation_between_sentences() -> None:
    """Multiple sentences: punctuation correctly interpolated."""
    from transcribe.models.asr.qwen3_asr import _align_text_to_timestamps

    TS = types.SimpleNamespace
    # text: "你好。我是。" time_stamps: "你好我是"
    time_stamps = [
        TS(text="你", start_time=0.0, end_time=0.2),
        TS(text="好", start_time=0.2, end_time=0.4),
        TS(text="我", start_time=1.0, end_time=1.2),
        TS(text="是", start_time=1.2, end_time=1.4),
    ]
    result = _align_text_to_timestamps("你好。我是。", time_stamps, offset=0.0)
    assert len(result) == 6
    # 你, 好, 。, 我, 是, 。
    assert result[0][0] == "你"
    assert result[1][0] == "好"
    assert result[2][0] == "。"
    assert result[3][0] == "我"
    assert result[4][0] == "是"
    assert result[5][0] == "。"


def test_transcribe_multi_sentence_preserves_punctuation() -> None:
    """transcribe() should produce multiple segments with punctuation."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    TS = types.SimpleNamespace
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    instance._context = ""
    instance._language = None

    mock_result = TS(
        text="你好。我是。",
        time_stamps=[
            TS(text="你", start_time=0.0, end_time=0.2),
            TS(text="好", start_time=0.2, end_time=0.4),
            TS(text="我", start_time=1.0, end_time=1.2),
            TS(text="是", start_time=1.2, end_time=1.4),
        ],
    )
    instance._model = types.SimpleNamespace(
        transcribe=lambda **kw: [mock_result]
    )

    audio = AudioSegment(
        waveform=np.zeros(32000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=2.0,
    )
    result = instance.transcribe(audio)
    assert len(result) == 2
    assert result[0].text == "你好"  # 。 discarded
    assert result[1].text == "我是"  # 。 discarded
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time == pytest.approx(0.42)  # 0.4 + 0.02 (punc interp)
    assert result[1].start_time == pytest.approx(1.0)
    assert result[1].end_time == pytest.approx(1.42)  # 1.4 + 0.02 (punc interp)
