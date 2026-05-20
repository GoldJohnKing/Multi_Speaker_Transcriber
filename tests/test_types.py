"""Tests for core data types."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import (
    AudioSegment,
    DiarizationResult,
    PipelineConfig,
    SpeakerSegment,
    TranscriptSegment,
)


# ---------------------------------------------------------------------------
# AudioSegment
# ---------------------------------------------------------------------------

class TestAudioSegment:
    def test_creation(self) -> None:
        waveform = np.zeros(16000, dtype=np.float32)
        seg = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            start_time=0.0,
            end_time=1.0,
        )
        assert seg.sample_rate == 16000
        assert seg.start_time == 0.0
        assert seg.end_time == 1.0
        np.testing.assert_array_equal(seg.waveform, waveform)

    def test_duration_property(self) -> None:
        seg = AudioSegment(
            waveform=np.zeros(16000),
            sample_rate=16000,
            start_time=1.5,
            end_time=3.5,
        )
        assert seg.duration == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SpeakerSegment
# ---------------------------------------------------------------------------

class TestSpeakerSegment:
    def test_creation(self) -> None:
        seg = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=2.5,
        )
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.start_time == 0.0
        assert seg.end_time == 2.5
        assert seg.is_overlap is False  # default

    def test_is_overlap_default_false(self) -> None:
        seg = SpeakerSegment(
            speaker_id="SPEAKER_01",
            start_time=0.0,
            end_time=1.0,
        )
        assert seg.is_overlap is False

    def test_is_overlap_explicit_true(self) -> None:
        seg = SpeakerSegment(
            speaker_id="SPEAKER_01",
            start_time=0.0,
            end_time=1.0,
            is_overlap=True,
        )
        assert seg.is_overlap is True

    def test_duration_property(self) -> None:
        seg = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=1.0,
            end_time=4.0,
        )
        assert seg.duration == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# DiarizationResult
# ---------------------------------------------------------------------------

class TestDiarizationResult:
    def test_creation_with_segments(self) -> None:
        segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 2.0),
            SpeakerSegment("SPEAKER_01", 2.0, 4.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)
        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.overlap_regions == []  # default

    def test_overlap_regions(self) -> None:
        result = DiarizationResult(
            segments=[],
            num_speakers=0,
            overlap_regions=[(1.0, 2.0), (5.0, 6.5)],
        )
        assert len(result.overlap_regions) == 2
        assert result.overlap_regions[0] == (1.0, 2.0)
        assert result.overlap_regions[1] == (5.0, 6.5)

    def test_default_overlap_regions(self) -> None:
        result = DiarizationResult(segments=[], num_speakers=0)
        assert result.overlap_regions == []

    def test_default_factory_isolation(self) -> None:
        """Two instances should not share the same overlap_regions list."""
        r1 = DiarizationResult(segments=[], num_speakers=0)
        r2 = DiarizationResult(segments=[], num_speakers=0)
        r1.overlap_regions.append((0.0, 1.0))
        assert r2.overlap_regions == []


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------

class TestTranscriptSegment:
    def test_creation(self) -> None:
        seg = TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=1.0,
            text="你好世界",
        )
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.start_time == 0.0
        assert seg.end_time == 1.0
        assert seg.text == "你好世界"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.device == "auto"
        assert cfg.diarize is True
        assert cfg.hotwords is None
        assert cfg.language == "zh"
        assert cfg.cache_dir == ".cache"
        assert cfg.num_speakers is None

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(
            device="cuda",
            diarize=False,
            hotwords="hotwords/dict.txt",
            language="en",
            cache_dir="/tmp/cache",
            num_speakers=3,
        )
        assert cfg.device == "cuda"
        assert cfg.diarize is False
        assert cfg.hotwords == "hotwords/dict.txt"
        assert cfg.language == "en"
        assert cfg.cache_dir == "/tmp/cache"
        assert cfg.num_speakers == 3
