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

    def test_per_speaker_independent_grouping(self) -> None:
        """Interleaved speakers produce independent per-speaker lines, not fragments.

        Speaker A says: 我(1.0) 觉(1.4) 得(1.8)
        Speaker B says: 不(1.2) 对(1.6)

        Linear grouping would give: A:我, B:不, A:觉, B:对, A:得 → 5 fragments.
        Per-speaker grouping gives: A:"我觉得" (1.0→1.8), B:"不对" (1.2→1.6) → 2 lines.
        """
        words = [
            WordTimestamp("我", 1.0, 1.1),
            WordTimestamp("不", 1.2, 1.3),
            WordTimestamp("觉", 1.4, 1.5),
            WordTimestamp("对", 1.6, 1.7),
            WordTimestamp("得", 1.8, 1.9),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=1.0,
                end_time=1.9,
                text="我觉得不对",
                is_overlap=False,
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.15),
                SpeakerSegment("SPEAKER_01", 1.15, 1.35),
                SpeakerSegment("SPEAKER_00", 1.35, 1.55),
                SpeakerSegment("SPEAKER_01", 1.55, 1.75),
                SpeakerSegment("SPEAKER_00", 1.75, 3.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)

        # Should produce 2 sub-segments (one per speaker), NOT 5 fragments
        assert len(result) == 2

        # SPEAKER_00 line: "我觉得" with time 1.0→1.9
        s00 = [s for s in result if s.speaker_id == "SPEAKER_00"]
        assert len(s00) == 1
        assert s00[0].text == "我觉得"
        assert s00[0].start_time == pytest.approx(1.0)
        assert s00[0].end_time == pytest.approx(1.9)
        assert s00[0].is_overlap is True

        # SPEAKER_01 line: "不对" with time 1.2→1.7
        s01 = [s for s in result if s.speaker_id == "SPEAKER_01"]
        assert len(s01) == 1
        assert s01[0].text == "不对"
        assert s01[0].start_time == pytest.approx(1.2)
        assert s01[0].end_time == pytest.approx(1.7)
        assert s01[0].is_overlap is True

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

    def test_speaker_with_gap_produces_separate_lines(self) -> None:
        """One speaker with a temporal gap in overlap produces separate lines."""
        words = [
            WordTimestamp("你", 1.0, 1.1),
            WordTimestamp("好", 1.1, 1.2),
            # 1.0s gap
            WordTimestamp("世", 2.2, 2.3),
            WordTimestamp("界", 2.3, 2.4),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=1.0,
                end_time=2.4,
                text="你好世界",
                is_overlap=False,
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
            ],
            num_speakers=1,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)

        # All words attributed to SPEAKER_00, but split into 2 lines by gap
        assert len(result) == 2
        assert result[0].text == "你好"
        assert result[0].start_time == pytest.approx(1.0)
        assert result[1].text == "世界"
        assert result[1].start_time == pytest.approx(2.2)
        assert all(s.speaker_id == "SPEAKER_00" for s in result)
        assert all(s.is_overlap is True for s in result)


def test_overlap_uses_non_exclusive_segments():
    """OverlapHandler should use non-exclusive segments for word attribution.

    In this scenario, exclusive diarization assigns the entire overlap region
    to SPEAKER_00, but non-exclusive shows both speakers are active.
    Without non-exclusive segments, all words would be attributed to SPEAKER_00.
    """
    words = [
        WordTimestamp("我", 5.0, 5.2),
        WordTimestamp("去", 5.2, 5.4),
        WordTimestamp("不", 5.4, 5.6),
        WordTimestamp("行", 5.6, 5.8),
    ]
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=5.0,
            end_time=5.8,
            text="我去不行",
            is_overlap=False,
            words=words,
        )
    ]
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 5.0, 5.8),
        ],
        num_speakers=2,
        overlap_regions=[(5.0, 5.8)],
        non_exclusive_segments=[
            SpeakerSegment("SPEAKER_00", 5.0, 5.4),
            SpeakerSegment("SPEAKER_01", 5.4, 5.8),
        ],
    )
    result = OverlapHandler().handle(segments, diarization)

    assert len(result) == 2
    s00 = [s for s in result if s.speaker_id == "SPEAKER_00"]
    s01 = [s for s in result if s.speaker_id == "SPEAKER_01"]
    assert len(s00) == 1
    assert s00[0].text == "我去"
    assert len(s01) == 1
    assert s01[0].text == "不行"


def test_short_segment_at_overlap_boundary():
    """Short segment mostly inside overlap should be detected.

    Old center-point check: center=5.0 NOT in [4.5, 5.0) -> missed.
    New intersection-ratio: 0.1/0.2 = 50% >= 50% -> detected.
    """
    words = [WordTimestamp("哇", 4.9, 5.1)]
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=4.9,
            end_time=5.1,
            text="哇",
            words=words,
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 4.0, 5.5)],
        num_speakers=1,
        overlap_regions=[(4.5, 5.0)],
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is True


def test_long_segment_barely_touching_overlap():
    """Segment with <50% overlap should NOT be treated as overlap."""
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            text="很长的一段话",
            words=[WordTimestamp("很", 0.0, 5.0)],
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
        num_speakers=1,
        overlap_regions=[(4.5, 5.0)],
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is False


def test_segment_fully_inside_overlap():
    """Segment fully inside overlap should always be detected."""
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=3.0,
            end_time=4.0,
            text="测试",
            words=[WordTimestamp("测", 3.0, 3.5), WordTimestamp("试", 3.5, 4.0)],
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
        num_speakers=1,
        overlap_regions=[(2.0, 5.0)],
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is True


def test_overlap_segments_have_zero_confidence():
    """Overlap-split segments should have attribution_confidence=0.0."""
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
    for seg in result:
        assert seg.attribution_confidence == 0.0


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
            overlap_regions=[(0.0, 0.3)],
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is True  # intersection=0.3 >= 0.4*0.5
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


class TestTurnBoundarySplitting:
    def test_segment_spanning_two_speakers_is_split(self):
        """A segment spanning A->B turn boundary should be split."""
        words = [
            WordTimestamp("大", 0.0, 0.5),
            WordTimestamp("家", 0.5, 1.0),
            WordTimestamp("好", 1.0, 1.5),
            WordTimestamp("谢", 2.0, 2.5),
            WordTimestamp("谢", 2.5, 3.0),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=3.0,
                text="大家好谢谢",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.8),
                SpeakerSegment("SPEAKER_01", 1.8, 4.0),
            ],
            num_speakers=2,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        # Should be split into two segments at the turn boundary (t=1.8)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "大家好"
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "谢谢"

    def test_no_turn_boundary_no_split(self):
        """Segment entirely within one speaker's turn should not be split."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
            num_speakers=1,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        assert len(result) == 1


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


class TestTurnBoundarySplittingWithoutWords:
    def test_segment_without_words_split_at_boundary(self):
        """Segment without word timestamps spanning a turn boundary should be
        split into sub-segments, each re-attributed to the correct speaker.
        Text is duplicated across sub-segments (known limitation).
        """
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=4.0,
                text="大家好谢谢",
            )
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

        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].start_time == pytest.approx(0.0)
        assert result[0].end_time == pytest.approx(2.0)
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].start_time == pytest.approx(2.0)
        assert result[1].end_time == pytest.approx(4.0)
        # Text is duplicated (known limitation: can't split text without words)
        assert result[0].text == "大家好谢谢"
        assert result[1].text == "大家好谢谢"
