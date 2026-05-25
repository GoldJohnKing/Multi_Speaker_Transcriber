"""Tests for the ASR backend modules."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp
from transcribe.models.asr import create_asr, list_backends, restore_hotwords, parse_timestamps
from transcribe.models.asr.utils import segment_by_timestamps
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.funasr_nano import FunASRNanoTranscriber
from transcribe.models.asr.funasr_paraformer import FunASRParaformerTranscriber


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


def test_list_backends_contains_all() -> None:
    """All four backends should be registered."""
    backends = list_backends()
    assert "Fun-ASR-Paraformer" in backends
    assert "Fun-ASR-Nano" in backends
    assert "Qwen3-ASR" in backends
    assert "Whisper" in backends


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


def test_default_transcribe():
    """Default transcribe() derives TranscriptSegment from transcribe_words()."""

    class MockASR(ASRBase):
        def __init__(self, device="cpu", hotword_path=None, **kwargs):
            pass

        def transcribe_words(self, audio):
            return [
                WordTimestamp(word="你好", start_time=0.0, end_time=1.0),
                WordTimestamp(word="世", start_time=1.0, end_time=1.2),
                WordTimestamp(word="界", start_time=1.2, end_time=1.5),
                WordTimestamp(word="测试", start_time=2.5, end_time=3.5),
            ]

        def cleanup(self):
            pass

    mock = MockASR()
    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000, start_time=0.0, end_time=1.0,
    )
    segments = mock.transcribe(audio)
    # "你好" + "世界" are < 0.5s gap → merged into one segment
    # "测试" is > 0.5s gap from "界" → separate segment
    assert len(segments) == 2
    assert segments[0].text == "你好世界"
    assert segments[0].start_time == pytest.approx(0.0)
    assert segments[0].end_time == pytest.approx(1.5)
    assert segments[1].text == "测试"
    assert segments[1].start_time == pytest.approx(2.5)


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


# --- segment_by_timestamps tests (pure function, no model needed) ---


def test_segment_empty_input() -> None:
    """Empty input returns empty list."""
    assert segment_by_timestamps([]) == []


def test_segment_single_char() -> None:
    """Single character produces one segment."""
    result = segment_by_timestamps([("你", 0.0, 0.2)])
    assert len(result) == 1
    assert result[0].text == "你"
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time == pytest.approx(0.2)


def test_segment_splits_at_sentence_end() -> None:
    """Segments split at sentence-ending punctuation (。！？)."""
    char_ts = [
        ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0),
        ("我", 2.0, 2.5), ("是", 2.5, 3.0), ("。", 3.0, 3.0),
    ]
    result = segment_by_timestamps(char_ts)
    assert len(result) == 2
    assert result[0].text == "你好"
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time == pytest.approx(1.0)
    assert result[1].text == "我是"
    assert result[1].start_time == pytest.approx(2.0)
    assert result[1].end_time == pytest.approx(3.0)


def test_segment_pause_splits_without_punctuation() -> None:
    """Long unpunctuated speech splits at pause (gap > max_gap_sec)."""
    char_ts = [(f"w{i}", float(i), float(i) + 0.5) for i in range(10)]
    result = segment_by_timestamps(char_ts, max_gap_sec=0.6)
    # All consecutive gaps are 0.5s (≤ 0.6) → single group
    assert len(result) == 1


def test_segment_pause_splits_at_large_gap() -> None:
    """Gaps > max_gap_sec produce splits."""
    char_ts = [
        ("你", 0.0, 0.5), ("好", 0.5, 1.0),
        ("世", 2.0, 2.5), ("界", 2.5, 3.0),
    ]
    result = segment_by_timestamps(char_ts, max_gap_sec=0.6)
    assert len(result) == 2
    assert result[0].text == "你好"
    assert result[1].text == "世界"


def test_segment_speaker_id_always_speaker_00() -> None:
    """All returned segments have speaker_id SPEAKER_00."""
    char_ts = [
        ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0),
        ("我", 2.0, 2.5), ("是", 2.5, 3.0), ("。", 3.0, 3.0),
    ]
    result = segment_by_timestamps(char_ts)
    for seg in result:
        assert seg.speaker_id == "SPEAKER_00"


def test_segment_preserves_time_offset() -> None:
    """Timestamps should preserve any offset in input (e.g. audio.start_time)."""
    offset = 10.5
    char_ts = [
        ("你", offset + 0.0, offset + 0.2),
        ("好", offset + 0.2, offset + 0.4),
        ("。", offset + 0.4, offset + 0.5),
    ]
    result = segment_by_timestamps(char_ts)
    assert result[0].start_time == pytest.approx(offset + 0.0)
    assert result[0].end_time == pytest.approx(offset + 0.4)  # content word end, not punctuation


class TestParaformerFallbackAlignment:
    def test_fallback_produces_per_char_words(self) -> None:
        """When exact alignment fails, fallback produces per-char words."""
        from transcribe.models.asr.funasr_paraformer import FunASRParaformerTranscriber

        # text has 3 non-punct chars + 1 punct = 4 chars total
        text = "你好，世"
        # _build_token_groups produces 3 non-punct tokens (你, 好, 世) + 1 punct (，)
        # 3 non-punct tokens != 4 timestamps → exact alignment fails → fallback
        timestamps = [[0, 100], [100, 200], [300, 400], [400, 500]]

        # Verify the mismatch that triggers fallback
        groups = FunASRParaformerTranscriber._build_token_groups(text)
        non_punct = [g for g in groups if not g[1]]
        assert len(non_punct) != len(timestamps)  # 3 != 4

        # Verify fallback produces per-char words
        fallback = FunASRParaformerTranscriber._fallback_char_timestamps(
            text, timestamps, audio_start=0.0,
        )
        assert len(fallback) == len(text)  # one WordTimestamp per char
        for w in fallback:
            assert w.end_time > w.start_time
