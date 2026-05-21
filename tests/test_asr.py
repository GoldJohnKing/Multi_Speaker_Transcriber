"""Tests for the ASR backend modules."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import create_asr, list_backends, restore_hotwords, parse_timestamps
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.nano import FunASRNanoTranscriber
from transcribe.models.asr.paraformer import FunASRParaformerTranscriber


@pytest.fixture
def silence_audio() -> AudioSegment:
    """1 second of silence at 16 kHz."""
    return AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )


# --- Factory tests ---


def test_list_backends_contains_both() -> None:
    """Both backends should be registered."""
    backends = list_backends()
    assert "Fun-ASR-Paraformer" in backends
    assert "Fun-ASR-Nano" in backends


def test_create_asr_unknown_backend_raises() -> None:
    """Unknown backend name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        create_asr("nonexistent")


@pytest.mark.slow
def test_create_asr_returns_correct_type() -> None:
    """create_asr should return the correct backend class."""
    nano = create_asr("Fun-ASR-Nano", device="cpu")
    assert isinstance(nano, FunASRNanoTranscriber)
    assert isinstance(nano, ASRBase)

    paraformer = create_asr("Fun-ASR-Paraformer", device="cpu")
    assert isinstance(paraformer, FunASRParaformerTranscriber)
    assert isinstance(paraformer, ASRBase)


@pytest.mark.slow
def test_backends_support_hotwords() -> None:
    """Both backends should declare hotword support."""
    nano = create_asr("Fun-ASR-Nano", device="cpu")
    paraformer = create_asr("Fun-ASR-Paraformer", device="cpu")
    assert nano.supports_hotwords is True
    assert paraformer.supports_hotwords is True


# --- FunASRNanoTranscriber slow tests ---


@pytest.mark.slow
def test_nano_init() -> None:
    """FunASRNanoTranscriber should initialise without error."""
    transcriber = FunASRNanoTranscriber(device="cpu")
    assert transcriber is not None
    assert transcriber._hotword_list == []


@pytest.mark.slow
def test_nano_transcribe_silence(silence_audio: AudioSegment) -> None:
    """Silence should produce empty or minimal output, not crash."""
    transcriber = FunASRNanoTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


@pytest.mark.slow
def test_nano_transcribe_with_hotwords(silence_audio: AudioSegment, tmp_path) -> None:
    """FunASRNanoTranscriber should accept a hotword file without error."""
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n", encoding="utf-8")
    transcriber = FunASRNanoTranscriber(device="cpu", hotword_path=str(hw_file))
    assert transcriber._hotword_list == ["张三", "李四"]
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)


@pytest.mark.slow
def test_nano_load_hotwords_missing_file() -> None:
    """Missing hotword file should be treated as no hotwords."""
    transcriber = FunASRNanoTranscriber(device="cpu", hotword_path="/nonexistent/path.txt")
    assert transcriber._hotword_list == []


# --- FunASRParaformerTranscriber slow tests ---


@pytest.mark.slow
def test_paraformer_init() -> None:
    """FunASRParaformerTranscriber should initialise without error."""
    transcriber = FunASRParaformerTranscriber(device="cpu")
    assert transcriber is not None
    assert transcriber._hotwords is None
    assert transcriber._hotword_list == []


@pytest.mark.slow
def test_paraformer_transcribe_silence(silence_audio: AudioSegment) -> None:
    """Silence should produce empty or minimal output, not crash."""
    transcriber = FunASRParaformerTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


@pytest.mark.slow
def test_paraformer_transcribe_with_hotwords(silence_audio: AudioSegment, tmp_path) -> None:
    """FunASRParaformerTranscriber should accept a hotword file without error."""
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n", encoding="utf-8")
    transcriber = FunASRParaformerTranscriber(device="cpu", hotword_path=str(hw_file))
    assert transcriber._hotwords == "张三 李四"
    assert transcriber._hotword_list == ["张三", "李四"]
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)


@pytest.mark.slow
def test_paraformer_load_hotwords_missing_file() -> None:
    """Missing hotword file should be treated as no hotwords."""
    transcriber = FunASRParaformerTranscriber(device="cpu", hotword_path="/nonexistent/path.txt")
    assert transcriber._hotwords is None
    assert transcriber._hotword_list == []


# --- Hotword punctuation restoration tests (pure function, no model needed) ---


def test_restore_removes_comma_inside_term() -> None:
    """Comma inserted inside a hotword term should be removed."""
    assert restore_hotwords("大家看朽，叶飞过来了", ["朽叶"]) == "大家看朽叶飞过来了"


def test_restore_preserves_external_punctuation() -> None:
    """Punctuation before/after a hotword should be preserved."""
    assert restore_hotwords("朽，叶，来了", ["朽叶"]) == "朽叶，来了"


def test_restore_multiple_hotwords() -> None:
    """Multiple hotwords with internal punctuation should all be restored."""
    result = restore_hotwords("硅，基流动的朽。叶团队", ["硅基流动", "朽叶"])
    assert result == "硅基流动的朽叶团队"


def test_restore_no_hotwords() -> None:
    """With no hotwords, text should pass through unchanged."""
    assert restore_hotwords("朽，叶来了", []) == "朽，叶来了"


def test_restore_single_char_hotword() -> None:
    """Single-character hotwords can't have internal punctuation, pass through."""
    assert restore_hotwords("好，的", ["好"]) == "好，的"


def test_restore_preserves_unrelated_text() -> None:
    """Text that doesn't match any hotword should be untouched."""
    assert restore_hotwords("今天天气很好，我们出去玩", ["硅基流动"]) == "今天天气很好，我们出去玩"


def test_restore_empty_string() -> None:
    """Empty string should return empty string."""
    assert restore_hotwords("", ["朽叶"]) == ""


def test_restore_various_punctuation() -> None:
    """Various Chinese punctuation types inside hotword should all be removed."""
    assert restore_hotwords("硅基？流动", ["硅基流动"]) == "硅基流动"


def test_restore_period_inside_term() -> None:
    """Period inserted inside a hotword should be removed."""
    assert restore_hotwords("来到了硅。基流动公司", ["硅基流动"]) == "来到了硅基流动公司"


def test_restore_two_char_hotword_with_space() -> None:
    """Space inside a hotword should be removed."""
    assert restore_hotwords("朽 叶", ["朽叶"]) == "朽叶"


def test_restore_hotword_at_string_boundaries() -> None:
    """Hotword at the start/end of text should be restored."""
    assert restore_hotwords("朽，叶来了", ["朽叶"]) == "朽叶来了"
    assert restore_hotwords("去找朽，叶", ["朽叶"]) == "去找朽叶"


def test_restore_repeated_hotword() -> None:
    """Multiple occurrences of the same hotword should all be restored."""
    assert restore_hotwords("朽，叶和朽。叶", ["朽叶"]) == "朽叶和朽叶"


def test_restore_overlapping_text_not_hotword() -> None:
    """Text containing hotword chars in different order should not be touched."""
    assert restore_hotwords("叶朽来了", ["朽叶"]) == "叶朽来了"


# --- Timestamp parsing tests (pure function, no model needed) ---


def test_parse_timestamps_dict_format() -> None:
    """Fun-ASR-Nano dict-format timestamps: seconds, first start / last end."""
    timestamps = [
        {"token": "你", "start_time": 0.10, "end_time": 0.20, "score": 0.99},
        {"token": "好", "start_time": 0.20, "end_time": 0.30, "score": 0.98},
    ]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.10)
    assert end == pytest.approx(0.30)


def test_parse_timestamps_single_entry() -> None:
    """Single-entry timestamp list."""
    timestamps = [
        {"token": "嗨", "start_time": 0.50, "end_time": 0.60, "score": 0.95},
    ]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.50)
    assert end == pytest.approx(0.60)


def test_parse_timestamps_empty_returns_none() -> None:
    """Empty timestamp list returns (None, None)."""
    start, end = parse_timestamps([])
    assert start is None
    assert end is None


def test_parse_timestamps_nested_list_format() -> None:
    """Legacy Paraformer format [[start_ms, end_ms], ...] should still be handled."""
    timestamps = [[100, 200], [200, 350]]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.100)
    assert end == pytest.approx(0.350)


def test_parse_timestamps_flat_list_format() -> None:
    """Legacy flat format [start, end, start, end, ...] should still be handled."""
    timestamps = [100, 200, 200, 350]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.100)
    assert end == pytest.approx(0.350)


def test_parse_timestamps_dict_missing_keys_returns_none() -> None:
    """Dict entries without start_time/end_time keys should not crash."""
    timestamps = [{"token": "<sil>"}]
    start, end = parse_timestamps(timestamps)
    assert start is None
    assert end is None
