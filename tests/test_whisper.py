"""Tests for the Whisper ASR backend."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import list_backends
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import _BACKENDS


# --- Registration tests ---


def test_whisper_registered() -> None:
    """"Whisper" should appear in the backend registry."""
    assert "Whisper" in list_backends()


def test_whisper_is_asr_base_subclass() -> None:
    """The registered class should be an ASRBase subclass."""
    from transcribe.models.asr.whisper import WhisperTranscriber

    assert issubclass(WhisperTranscriber, ASRBase)


def test_supports_hotwords_true() -> None:
    """Whisper should declare hotword support."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    assert instance.supports_hotwords is True


# --- Import error test ---


def test_import_error_without_faster_whisper() -> None:
    """Instantiation without faster-whisper installed should give a helpful error."""
    from transcribe.models.asr.whisper import WhisperTranscriber

    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(ImportError, match="uv sync --extra whisper"):
            WhisperTranscriber(device="cpu")


# --- Hotword loading tests ---


def test_load_hotwords_reads_file(tmp_path) -> None:
    """_load_hotwords should read hotword file and comma-join."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)

    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n王五\n", encoding="utf-8")

    result = instance._load_hotwords(str(hw_file))
    assert result == "张三,李四,王五"


def test_load_hotwords_none_returns_none() -> None:
    """_load_hotwords with None path returns None."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    assert instance._load_hotwords(None) is None


def test_load_hotwords_empty_file_returns_none(tmp_path) -> None:
    """_load_hotwords with empty file returns None."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)

    hw_file = tmp_path / "empty.txt"
    hw_file.write_text("", encoding="utf-8")

    result = instance._load_hotwords(str(hw_file))
    assert result is None


def test_load_hotwords_missing_file_returns_none() -> None:
    """_load_hotwords with missing file returns None (no crash)."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    result = instance._load_hotwords("/nonexistent/path/hotwords.txt")
    assert result is None


# --- Transcribe tests with mocked model ---


def _make_instance() -> types.SimpleNamespace:
    """Create a WhisperTranscriber instance via __new__ with mocked internals."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    instance._device = "cpu"
    instance._language = None
    instance._hotwords = None
    instance._model = MagicMock()
    return instance


def _make_segment(
    start: float, end: float, text: str
) -> types.SimpleNamespace:
    """Create a mock faster-whisper Segment."""
    return types.SimpleNamespace(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[],
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=None,
        temperature=0.0,
    )


def _make_info(language: str = "zh") -> types.SimpleNamespace:
    """Create a mock faster-whisper TranscriptionInfo."""
    return types.SimpleNamespace(
        language=language,
        language_probability=0.99,
        duration=2.0,
        duration_after_vad=2.0,
        all_language_probs=None,
        transcription_options=None,
        vad_options=None,
    )


def test_transcribe_single_segment() -> None:
    """transcribe() should map a single segment to TranscriptSegment."""
    instance = _make_instance()
    segment = _make_segment(0.5, 1.5, "你好世界")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=16000,
        start_time=10.0,
        end_time=12.0,
    )
    result = instance.transcribe(audio)

    assert len(result) == 1
    assert result[0].text == "你好世界"
    assert result[0].start_time == pytest.approx(10.5)  # 0.5 + 10.0 offset
    assert result[0].end_time == pytest.approx(11.5)  # 1.5 + 10.0 offset
    assert result[0].speaker_id == "SPEAKER_00"


def test_transcribe_multiple_segments() -> None:
    """transcribe() should map multiple segments preserving order."""
    instance = _make_instance()
    seg1 = _make_segment(0.0, 1.0, "第一段")
    seg2 = _make_segment(1.5, 2.5, "第二段")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([seg1, seg2]), info)

    audio = AudioSegment(
        waveform=np.zeros(40000, dtype=np.float32),
        sample_rate=16000,
        start_time=5.0,
        end_time=7.5,
    )
    result = instance.transcribe(audio)

    assert len(result) == 2
    assert result[0].text == "第一段"
    assert result[0].start_time == pytest.approx(5.0)
    assert result[1].text == "第二段"
    assert result[1].start_time == pytest.approx(6.5)


def test_transcribe_empty_segments() -> None:
    """transcribe() returns empty list when model produces no segments."""
    instance = _make_instance()
    info = _make_info()

    instance._model.transcribe.return_value = (iter([]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)
    assert result == []


def test_transcribe_skips_empty_text_segments() -> None:
    """transcribe() should skip segments with empty text."""
    instance = _make_instance()
    seg1 = _make_segment(0.0, 0.5, "")
    seg2 = _make_segment(0.5, 1.5, "有效内容")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([seg1, seg2]), info)

    audio = AudioSegment(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.5,
    )
    result = instance.transcribe(audio)

    assert len(result) == 1
    assert result[0].text == "有效内容"


def test_transcribe_strips_whitespace() -> None:
    """transcribe() should strip leading/trailing whitespace from text."""
    instance = _make_instance()
    segment = _make_segment(0.0, 1.0, "  你好  ")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)

    assert result[0].text == "你好"


def test_transcribe_passes_hotwords() -> None:
    """transcribe() should pass hotwords to model.transcribe()."""
    instance = _make_instance()
    instance._hotwords = "张三,李四"
    segment = _make_segment(0.0, 1.0, "你好")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    instance.transcribe(audio)

    call_kwargs = instance._model.transcribe.call_args
    assert call_kwargs[1]["hotwords"] == "张三,李四"


def test_transcribe_zero_offset() -> None:
    """transcribe() with start_time=0 should use segment timestamps directly."""
    instance = _make_instance()
    segment = _make_segment(0.36, 0.89, "测试")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)

    assert result[0].start_time == pytest.approx(0.36)
    assert result[0].end_time == pytest.approx(0.89)


def test_transcribe_passes_language() -> None:
    """transcribe() should pass language to model.transcribe()."""
    instance = _make_instance()
    instance._language = "en"
    segment = _make_segment(0.0, 1.0, "hello")
    info = _make_info(language="en")

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    instance.transcribe(audio)

    call_kwargs = instance._model.transcribe.call_args
    assert call_kwargs[1]["language"] == "en"


# --- Cleanup tests ---


def test_cleanup_deletes_model() -> None:
    """cleanup() should delete _model attribute."""
    instance = _make_instance()
    assert hasattr(instance, "_model")
    instance.cleanup()
    assert not hasattr(instance, "_model")
