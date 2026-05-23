"""Tests for the speaker attribution module."""
from __future__ import annotations

import pytest

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)
from transcribe.models.attribution.engine import AttributionEngine
from transcribe.models.attribution.overlap import OverlapHandler
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


class TestOverlapHandler:
    def test_no_overlap_regions_returns_unchanged(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].is_overlap is False

    def test_overlap_no_words_marks_flag(self) -> None:
        """Degraded path: segment in overlap but no word timestamps."""
        segments = [TranscriptSegment("SPEAKER_00", 1.0, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 0.0, 3.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].is_overlap is True

    def test_overlap_splits_by_speaker(self) -> None:
        """Two speakers overlap; words attributed and split into sub-segments."""
        words = [
            WordTimestamp("这", 10.0, 10.1),
            WordTimestamp("个", 10.1, 10.2),
            WordTimestamp("对", 10.3, 10.4),
            WordTimestamp("好", 10.5, 10.6),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=10.0,
                end_time=10.6,
                text="这个对好",
                is_overlap=False,
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 10.0, 10.3),
                SpeakerSegment("SPEAKER_01", 10.3, 10.7),
            ],
            num_speakers=2,
            overlap_regions=[(10.0, 10.6)],
        )
        result = OverlapHandler().handle(segments, diarization)

        # Should produce 2 sub-segments (one per speaker)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "这个"
        assert result[0].start_time == pytest.approx(10.0)
        assert result[0].end_time == pytest.approx(10.2)
        assert result[0].is_overlap is True

        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "对好"
        assert result[1].start_time == pytest.approx(10.3)
        assert result[1].end_time == pytest.approx(10.6)
        assert result[1].is_overlap is True

    def test_non_overlap_segment_passes_through(self) -> None:
        """Segment outside overlap region is unchanged."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 3.0, 4.0, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
            num_speakers=1,
            overlap_regions=[(1.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 2
        assert all(s.is_overlap is False for s in result)

    def test_zero_duration_word_attribution(self) -> None:
        """Zero-duration word should be attributed by midpoint, not default SPEAKER_00."""
        words = [
            WordTimestamp("我", 5.0, 5.2),
            WordTimestamp("了", 5.2, 5.2),  # zero-duration
            WordTimestamp("的", 5.3, 5.5),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=5.0,
                end_time=5.5,
                text="我了的",
                is_overlap=False,
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 5.1),
                SpeakerSegment("SPEAKER_01", 5.1, 10.0),
            ],
            num_speakers=2,
            overlap_regions=[(4.0, 6.0)],
        )
        result = OverlapHandler().handle(segments, diarization)

        # "了" at 5.2 should be attributed to SPEAKER_01 (nearest by midpoint),
        # not hardcoded to SPEAKER_00
        assert len(result) >= 1
        speakers = {seg.speaker_id for seg in result}
        assert "SPEAKER_01" in speakers

    def test_all_words_same_speaker_produces_one_segment(self) -> None:
        """All words in overlap attributed to same speaker → one sub-segment."""
        words = [
            WordTimestamp("你", 1.0, 1.1),
            WordTimestamp("好", 1.1, 1.2),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=1.0,
                end_time=1.2,
                text="你好",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 0.0, 0.5),
            ],
            num_speakers=2,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is True


class TestAttributionEngine:
    def test_full_attribution_flow(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 0.4, "你好"),
            TranscriptSegment("SPEAKER_00", 1.0, 1.4, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.8),
                SpeakerSegment("SPEAKER_01", 0.8, 2.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.1, 0.3)],
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is True  # center=0.2 in [0.1, 0.3)
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "世界"
        assert result[1].is_overlap is False

    def test_no_overlap_regions(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 0.5, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        assert result[0].is_overlap is False

    def test_overlap_with_word_splitting(self) -> None:
        """Integration: engine splits overlap by word-level attribution."""
        words = [
            WordTimestamp("这", 10.0, 10.1),
            WordTimestamp("个", 10.1, 10.2),
            WordTimestamp("好", 10.3, 10.4),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=10.0,
                end_time=10.4,
                text="这个好",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 10.0, 10.25),
                SpeakerSegment("SPEAKER_01", 10.25, 10.5),
            ],
            num_speakers=2,
            overlap_regions=[(10.0, 10.4)],
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        # Should split into 2 sub-segments
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "这个"
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "好"
