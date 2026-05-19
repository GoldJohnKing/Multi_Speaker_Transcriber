"""Tests for the ASR transcriber module."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import ASRTranscriber, restore_hotwords


@pytest.fixture
def silence_audio() -> AudioSegment:
    """1 second of silence at 16 kHz."""
    return AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )


def test_transcriber_init() -> None:
    """ASRTranscriber should initialise without error."""
    transcriber = ASRTranscriber(device="cpu")
    assert transcriber is not None


def test_transcribe_silence(silence_audio: AudioSegment) -> None:
    """Silence should produce empty or minimal output, not crash."""
    transcriber = ASRTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


def test_transcribe_with_hotwords(silence_audio: AudioSegment, tmp_path: pytest.TempPathFactory) -> None:
    """ASRTranscriber should accept a hotword file without error."""
    hw_file = tmp_path / "hotwords.txt"  # type: ignore[operator]
    hw_file.write_text("张三\n李四\n", encoding="utf-8")
    transcriber = ASRTranscriber(device="cpu", hotword_path=str(hw_file))
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)


def test_load_hotwords_missing_file() -> None:
    """Missing hotword file should be treated as no hotwords."""
    transcriber = ASRTranscriber(device="cpu", hotword_path="/nonexistent/path.txt")
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


def test_restore_no_hotwords_none_list() -> None:
    """With empty hotword list, text passes through."""
    assert restore_hotwords("朽，叶来了", []) == "朽，叶来了"


def test_restore_single_char_hotword() -> None:
    """Single-character hotwords can't have internal punctuation, pass through."""
    # "好" is single char — no restoration pattern built, text unchanged
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
