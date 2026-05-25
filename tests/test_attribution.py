"""Tests for the speaker attribution module."""
from __future__ import annotations

import pytest

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
)
from transcribe.models.attribution.engine import AttributionEngine
from transcribe.models.attribution.strategy import TimestampStrategy


class TestTimestampStrategyBasic:
    def test_single_segment_single_speaker(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.5, 2.0, "你好世界"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好世界"

    def test_two_segments_two_speakers(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.5),
                SpeakerSegment("SPEAKER_01", 1.5, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_dominant_speaker_by_intersection(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.5),
                SpeakerSegment("SPEAKER_01", 1.5, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].speaker_id == "SPEAKER_00"

    def test_fallback_to_nearest_when_no_overlap(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 1.2, 1.3, "嗯"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 2.0, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"

    def test_empty_segments(self) -> None:
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute([], diarization)
        assert result == []

    def test_empty_diarization(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好")]
        diarization = DiarizationResult(segments=[], num_speakers=0)
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].speaker_id == "SPEAKER_00"

    def test_multiple_segments_preserved(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 1.0, 2.0, "世界"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "再见"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 3
        assert [s.text for s in result] == ["你好", "世界", "再见"]

    def test_timing_preserved(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.5, 1.5, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].start_time == pytest.approx(0.5)
        assert result[0].end_time == pytest.approx(1.5)


class TestAttributionEngine:
    def test_engine_delegates_to_strategy(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.5),
                SpeakerSegment("SPEAKER_01", 1.5, 3.0),
            ],
            num_speakers=2,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_engine_no_splitting(self) -> None:
        """Engine does not split segments — only assigns speakers."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 4.0, "跨说话人的长句子"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 2.0),
                SpeakerSegment("SPEAKER_01", 2.0, 4.0),
            ],
            num_speakers=2,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        # Segment is NOT split — just attributed to dominant speaker
        assert len(result) == 1
        assert result[0].text == "跨说话人的长句子"


class TestAttributionConfidence:
    def test_single_speaker_high_confidence(self):
        """Segment fully within one speaker -> confidence 1.0."""
        segments = [TranscriptSegment("SPEAKER_00", 0.5, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == 1.0

    def test_equal_overlap_low_confidence(self):
        """Equal overlap with two speakers -> confidence ~0.5."""
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 1.0, 2.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert 0.45 <= result[0].attribution_confidence <= 0.55

    def test_fallback_zero_confidence(self):
        """No overlap, nearest-speaker fallback -> confidence 0.0."""
        segments = [TranscriptSegment("SPEAKER_00", 1.2, 1.3, "嗯")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 2.0, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == 0.0

    def test_dominant_speaker_high_confidence(self):
        """75/25 split -> confidence ~0.75."""
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 4.0, "你好世界")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 3.0, 4.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == pytest.approx(0.75, abs=0.01)
