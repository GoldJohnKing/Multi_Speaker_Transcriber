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
from transcribe.models.attribution.overlap import (
    MarkOverlapHandler,
    OverlapHandler,
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


class TestTimestampStrategyMultiSmoothing:
    def test_consecutive_short_interrupts_merged(self) -> None:
        """Multiple consecutive short segments from same speaker are smoothed."""
        words = [
            WordTimestamp("你好", 0.0, 0.4),
            WordTimestamp("嗯", 0.5, 0.55),     # short, SPEAKER_01
            WordTimestamp("啊", 0.6, 0.65),     # short, SPEAKER_01
            WordTimestamp("天气", 0.7, 1.0),    # back to SPEAKER_00
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.45),
                SpeakerSegment("SPEAKER_01", 0.45, 0.575),
                SpeakerSegment("SPEAKER_01", 0.575, 0.675),
                SpeakerSegment("SPEAKER_00", 0.675, 1.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy(min_segment_duration=0.2).attribute(words, diarization)
        # Both "嗯" and "啊" (< 0.2s each) should be smoothed into SPEAKER_00
        for seg in result:
            assert seg.speaker_id == "SPEAKER_00"


class TestMarkOverlapHandler:
    def test_segment_in_overlap_region_marked(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 1.0, 2.0, "你好"),
            TranscriptSegment("SPEAKER_01", 3.0, 4.0, "世界"),
        ]
        overlap_regions = [(0.5, 2.5)]
        result = MarkOverlapHandler().process(segments, overlap_regions)
        assert result[0].is_overlap is True
        assert result[1].is_overlap is False

    def test_no_overlap_regions(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
        ]
        result = MarkOverlapHandler().process(segments, [])
        assert result[0].is_overlap is False

    def test_partial_overlap(self) -> None:
        """Segment whose center is in an overlap region is marked."""
        segments = [
            TranscriptSegment("SPEAKER_00", 1.0, 3.0, "你好"),  # center=2.0
        ]
        overlap_regions = [(1.5, 2.5)]
        result = MarkOverlapHandler().process(segments, overlap_regions)
        assert result[0].is_overlap is True

    def test_edge_touch_not_marked(self) -> None:
        """Segment only touching overlap edge is NOT marked (center outside)."""
        segments = [
            TranscriptSegment("SPEAKER_00", 1.8, 3.0, "你好"),  # center=2.4
        ]
        overlap_regions = [(1.0, 2.0)]  # center 2.4 is outside [1.0, 2.0)
        result = MarkOverlapHandler().process(segments, overlap_regions)
        assert result[0].is_overlap is False


class TestAttributionEngine:
    def test_full_attribution_flow(self) -> None:
        """End-to-end: words + diarization + overlap → attributed segments."""
        words = [
            WordTimestamp("你", 0.0, 0.2),
            WordTimestamp("好", 0.2, 0.4),
            WordTimestamp("世", 1.0, 1.2),
            WordTimestamp("界", 1.2, 1.4),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.8),
                SpeakerSegment("SPEAKER_01", 0.8, 2.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.3, 0.5)],
        )
        engine = AttributionEngine()
        result = engine.run(words, diarization, diarization.overlap_regions)

        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is False  # center 0.2 outside [0.3, 0.5)
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "世界"
        assert result[1].is_overlap is False

    def test_no_overlap_regions(self) -> None:
        words = [WordTimestamp("你好", 0.0, 0.5)]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
            overlap_regions=[],
        )
        engine = AttributionEngine()
        result = engine.run(words, diarization, [])
        assert result[0].is_overlap is False


class TestPunctuationAttribution:
    def test_punctuation_follows_preceding_speaker(self) -> None:
        """Punctuation at a speaker boundary follows the preceding word's speaker."""
        words = [
            WordTimestamp("你", 0.0, 0.2),
            WordTimestamp("好", 0.2, 0.4),
            WordTimestamp("。", 0.49, 0.51),  # center 0.5 → straddles boundary
            WordTimestamp("世", 1.0, 1.2),
            WordTimestamp("界", 1.2, 1.4),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.5),
                SpeakerSegment("SPEAKER_01", 0.5, 2.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(words, diarization)
        # Without fix: "。" (center=0.5) falls in SPEAKER_01 → "世界。"
        # With fix:    "。" follows "好" → SPEAKER_00 → "你好。"
        assert result[0].speaker_id == "SPEAKER_00"
        assert "。" in result[0].text

    def test_punctuation_at_start_follows_next_word(self) -> None:
        """Leading punctuation follows the next non-punct word's speaker."""
        words = [
            WordTimestamp("。", 0.49, 0.51),
            WordTimestamp("世", 1.0, 1.2),
            WordTimestamp("界", 1.2, 1.4),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.5),
                SpeakerSegment("SPEAKER_01", 0.5, 2.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(words, diarization)
        # Leading "。" should follow next word "世" → SPEAKER_01
        assert result[0].speaker_id == "SPEAKER_01"
