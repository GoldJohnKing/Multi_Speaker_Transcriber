"""Tests for the speaker attribution module."""
from __future__ import annotations

import pytest

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
)
from transcribe.models.attribution.engine import AttributionEngine
from transcribe.models.attribution.overlap import MarkOverlapHandler
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


class TestMarkOverlapHandler:
    def test_segment_in_overlap_region_marked(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 1.0, 2.0, "你好"),
            TranscriptSegment("SPEAKER_01", 3.0, 4.0, "世界"),
        ]
        result = MarkOverlapHandler().process(segments, [(0.5, 2.5)])
        assert result[0].is_overlap is True
        assert result[1].is_overlap is False

    def test_no_overlap_regions(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好")]
        result = MarkOverlapHandler().process(segments, [])
        assert result[0].is_overlap is False

    def test_partial_overlap(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 1.0, 3.0, "你好")]
        result = MarkOverlapHandler().process(segments, [(1.5, 2.5)])
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
        result = engine.run(segments, diarization, diarization.overlap_regions)

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
        result = engine.run(segments, diarization, [])
        assert result[0].is_overlap is False
