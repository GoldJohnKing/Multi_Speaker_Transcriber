"""Tests for transcribe.models.segmentation.SubtitleSegmenter.

The segmenter uses a single-pass ``punctuation_or_pause`` algorithm:
split when the last buffered word is a sentence-ending punctuation mark,
or when the inter-word gap exceeds *max_gap_sec*.
"""

from __future__ import annotations

import pytest

from transcribe.data.types import TranscriptSegment, WordTimestamp
from transcribe.models.segmentation import SubtitleSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _w(chars: str, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(chars, start, end)


def _ws(spec: list[tuple[str, float, float]]) -> list[WordTimestamp]:
    return [_w(c, s, e) for c, s, e in spec]


def _texts(segments: list[TranscriptSegment]) -> list[str]:
    return [s.text for s in segments]


# ---------------------------------------------------------------------------
# TestPunctuationSplit
# ---------------------------------------------------------------------------


class TestPunctuationSplit:
    """Sentence-ending punctuation triggers a split; ！？ are retained in output."""

    def test_single_sentence_no_punctuation(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_split_at_period(self) -> None:
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02),
            ("世", 1.5, 2.0), ("界", 2.0, 2.5),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好", "世界"]

    def test_split_at_exclamation(self) -> None:
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.02), ("棒", 1.5, 2.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！", "棒"]

    def test_split_at_question(self) -> None:
        words = _ws([("是", 0.0, 0.5), ("吗", 0.5, 1.0), ("？", 1.0, 1.02), ("对", 1.5, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["是吗？", "对"]

    def test_multiple_sentence_end(self) -> None:
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02),
            ("世", 1.5, 2.0), ("界", 2.0, 2.5), ("！", 2.5, 2.52),
            ("再", 3.0, 3.5), ("见", 3.5, 4.0),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好", "世界！", "再见"]

    def test_trailing_sentence_end_no_empty(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_trailing_exclamation_retained(self) -> None:
        """Trailing ！ is retained in the final segment text."""
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.02)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！"]

    def test_consecutive_sentence_end_marks(self) -> None:
        """Consecutive ！？ — ！ attaches to preceding content, ？ is stripped."""
        words = _ws([
            ("好", 0.0, 1.0), ("！", 1.0, 1.02), ("？", 1.02, 1.04), ("棒", 1.5, 2.5),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！", "棒"]

    def test_leading_sentence_end_discarded(self) -> None:
        words = _ws([("！", 0.0, 0.02), ("你", 0.5, 1.0), ("好", 1.0, 1.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_empty_input(self) -> None:
        segs = SubtitleSegmenter().segment([])
        assert segs == []


# ---------------------------------------------------------------------------
# TestPauseSplit
# ---------------------------------------------------------------------------


class TestPauseSplit:
    """Inter-word gaps exceeding *max_gap_sec* trigger a split."""

    def test_no_gap_single_group(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 1.0, 1.5)])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert _texts(segs) == ["你好世"]

    def test_large_gap_splits(self) -> None:
        """Gap > max_gap_sec produces a split."""
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 2.5, 3.0)])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert len(segs) == 2
        assert _texts(segs) == ["你好", "世"]

    def test_small_gap_no_split(self) -> None:
        """Gap <= max_gap_sec does not split."""
        words = _ws([("你", 0.0, 0.5), ("好", 0.6, 1.1)])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert _texts(segs) == ["你好"]

    def test_gap_threshold_boundary(self) -> None:
        """Gap exactly equal to max_gap_sec does NOT split (uses >)."""
        words = _ws([("你", 0.0, 0.5), ("好", 1.1, 1.6)])  # gap ≈ 0.6
        segs = SubtitleSegmenter(max_gap_sec=0.7).segment(words)
        assert _texts(segs) == ["你好"]

    def test_multiple_gaps(self) -> None:
        words = _ws([
            ("甲", 0.0, 0.5), ("乙", 0.5, 1.0),
            ("丙", 2.0, 2.5), ("丁", 2.5, 3.0),
            ("戊", 4.0, 4.5),
        ])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert _texts(segs) == ["甲乙", "丙丁", "戊"]

    def test_comma_no_false_gap(self) -> None:
        """Comma (near-zero duration) between words should not create a false gap."""
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert _texts(segs) == ["你好"]


# ---------------------------------------------------------------------------
# TestCombinedPunctuationAndPause
# ---------------------------------------------------------------------------


class TestCombinedPunctuationAndPause:
    """Punctuation and pause splitting work together."""

    def test_punctuation_and_pause_combined(self) -> None:
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02),
            ("世", 3.0, 3.5), ("界", 3.5, 4.0),
        ])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert _texts(segs) == ["你好", "世界"]

    def test_pause_creates_split_without_punctuation(self) -> None:
        """Long unpunctuated speech splits at pause."""
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 2.5, 3.0)])
        segs = SubtitleSegmenter(max_gap_sec=0.6).segment(words)
        assert len(segs) == 2


# ---------------------------------------------------------------------------
# TestOutputFormatting
# ---------------------------------------------------------------------------


class TestOutputFormatting:
    """Punctuation stripping and CJK-aware joining in output."""

    def test_period_stripped_from_output(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_exclamation_retained_in_output(self) -> None:
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.02)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！"]

    def test_question_mark_retained_in_output(self) -> None:
        words = _ws([("是", 0.0, 0.5), ("吗", 0.5, 1.0), ("？", 1.0, 1.02)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["是吗？"]

    def test_comma_stripped_from_output(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.52), ("好", 0.8, 1.3)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_cjk_no_space_joining(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert segs[0].text == "你好"

    def test_default_speaker_id(self) -> None:
        words = _ws([("你", 0.0, 0.5)])
        segs = SubtitleSegmenter().segment(words)
        assert segs[0].speaker_id == "SPEAKER_00"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and real-world scenarios."""

    def test_single_word(self) -> None:
        words = _ws([("好", 0.0, 0.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好"]

    def test_all_punctuation(self) -> None:
        words = _ws([("。", 0.0, 0.02), ("！", 0.02, 0.04), ("？", 0.04, 0.06)])
        segs = SubtitleSegmenter().segment(words)
        assert segs == []

    def test_only_comma_between_words(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.52), ("好", 0.52, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_segment_timing(self) -> None:
        words = _ws([("你", 1.5, 2.0), ("好", 2.0, 2.8)])
        segs = SubtitleSegmenter().segment(words)
        assert len(segs) == 1
        assert segs[0].start_time == 1.5
        assert segs[0].end_time == 2.8

    def test_realistic_chinese_conversation(self) -> None:
        words = _ws([
            ("我", 0.0, 0.3), ("们", 0.3, 0.6), ("今", 0.6, 0.9), ("天", 0.9, 1.2),
            ("去", 1.2, 1.5), ("了", 1.5, 1.8), ("公", 1.8, 2.1), ("园", 2.1, 2.4),
            ("。", 2.4, 2.42),
            ("天", 2.8, 3.1), ("气", 3.1, 3.4), ("很", 3.4, 3.7), ("好", 3.7, 4.0),
            ("。", 4.0, 4.02),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["我们今天去了公园", "天气很好"]

    def test_words_preserved_in_output(self) -> None:
        """Each segment's .words contains the content WordTimestamps."""
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.02), ("世", 1.5, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert len(segs) == 2
        # Period is stripped, so words only contain content words
        assert segs[0].words is not None
        assert [w.word for w in segs[0].words] == ["你", "好"]
        assert [w.word for w in segs[1].words] == ["世"]
