"""Tests for the speaker attribution module."""
from __future__ import annotations

import pytest

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    WordTimestamp,
)
from transcribe.models.attribution.strategy import TimestampStrategy


class TestTimestampStrategyBasic:
    def test_single_word_single_speaker(self) -> None:
        """One word fully within one speaker segment."""
        words = [WordTimestamp("你好", 0.5, 1.0)]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(words, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"

    def test_two_speakers_alternating(self) -> None:
        """Words from two speakers in turn."""
        words = [
            WordTimestamp("你好", 0.0, 0.5),
            WordTimestamp("世界", 1.5, 2.0),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 1.0, 2.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(words, diarization)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_merge_consecutive_same_speaker(self) -> None:
        """Consecutive words from the same speaker should merge."""
        words = [
            WordTimestamp("你", 0.0, 0.2),
            WordTimestamp("好", 0.2, 0.4),
            WordTimestamp("世", 0.4, 0.6),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(words, diarization)
        assert len(result) == 1
        assert result[0].text == "你好世"
        assert result[0].start_time == pytest.approx(0.0)
        assert result[0].end_time == pytest.approx(0.6)

    def test_empty_words(self) -> None:
        words = []
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(words, diarization)
        assert result == []

    def test_word_outside_diarization_falls_to_nearest(self) -> None:
        """Word between two diarization segments falls to nearest."""
        words = [WordTimestamp("嗯", 1.5, 1.6)]  # gap between segments
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 2.0, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(words, diarization)
        assert len(result) == 1
        # Center point 1.55 is closer to SPEAKER_01 start (2.0, distance 0.45)
        # than SPEAKER_00 end (1.0, distance 0.55)
        assert result[0].speaker_id == "SPEAKER_01"


class TestTimestampStrategySmoothing:
    def test_short_interrupt_merged(self) -> None:
        """Short interruption (< min_segment_duration) between same speaker merged."""
        words = [
            WordTimestamp("你好", 0.0, 0.4),
            WordTimestamp("嗯", 0.5, 0.6),   # 0.1s, very short
            WordTimestamp("天气", 0.7, 1.0),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.45),
                SpeakerSegment("SPEAKER_01", 0.45, 0.65),
                SpeakerSegment("SPEAKER_00", 0.65, 1.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy(min_segment_duration=0.2).attribute(words, diarization)
        # "嗯" (0.1s < 0.2s) should be smoothed into surrounding SPEAKER_00
        for seg in result:
            assert seg.speaker_id == "SPEAKER_00"
