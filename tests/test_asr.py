"""Tests for the ASR transcriber module."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import ASRTranscriber


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
