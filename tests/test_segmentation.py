"""Tests for transcribe.models.segmentation.SubtitleSegmenter."""

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
# TestSentenceEndSplit
# ---------------------------------------------------------------------------


class TestSentenceEndSplit:
    """Sentence-ending punctuation triggers a hard split and is discarded."""

    def test_single_sentence_no_punctuation(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_split_at_period(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0), ("世", 1.0, 1.5), ("界", 1.5, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好", "世界"]

    def test_split_at_exclamation(self) -> None:
        # Durations must exceed min_duration (0.833s) to avoid merge-back
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.0), ("棒", 1.0, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好", "棒"]

    def test_split_at_question(self) -> None:
        words = _ws([("是", 0.0, 0.5), ("吗", 0.5, 1.0), ("？", 1.0, 1.0), ("对", 1.0, 1.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["是吗", "对"]

    def test_multiple_sentence_end(self) -> None:
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0),
            ("世", 1.0, 1.5), ("界", 1.5, 2.0), ("！", 2.0, 2.0),
            ("再", 2.0, 2.5), ("见", 2.5, 3.0),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好", "世界", "再见"]

    def test_trailing_sentence_end_no_empty(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_leading_sentence_end_discarded(self) -> None:
        words = _ws([("！", 0.0, 0.0), ("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_empty_input(self) -> None:
        segs = SubtitleSegmenter().segment([])
        assert segs == []


# ---------------------------------------------------------------------------
# TestClauseSplit
# ---------------------------------------------------------------------------


class TestClauseSplit:
    """Clause punctuation is used for soft splits when limits are exceeded."""

    def _make_long_duration_words(self) -> list[WordTimestamp]:
        """Words spanning > 7 s with a comma in the middle."""
        return _ws([
            ("这", 0.0, 0.5), ("是", 0.5, 1.0),
            ("，", 1.0, 1.0),
            ("一", 1.0, 3.0), ("段", 3.0, 5.0), ("很", 5.0, 7.0), ("长", 7.0, 8.5),
        ])

    def test_clause_split_when_over_duration(self) -> None:
        words = self._make_long_duration_words()
        segs = SubtitleSegmenter().segment(words)
        # Total duration 8.5 > 7.0 → should split at comma
        assert len(segs) == 2
        assert _texts(segs) == ["这是", "一段很长"]

    def test_clause_split_when_over_max_chars(self) -> None:
        segs_list = "ABCDEFGHIJKLMNOP"  # 16 chars
        words = _ws([(c, i * 0.2, (i + 1) * 0.2) for i, c in enumerate(segs_list[:12])])
        # Insert a comma after 6th word
        words.insert(6, _w("，", 1.2, 1.2))
        seg = SubtitleSegmenter(max_chars=10)
        result = seg.segment(words)
        # Should split at comma because first part exceeds max_chars
        assert len(result) >= 2

    def test_hard_cut_no_clause(self) -> None:
        """When over limits with no clause punctuation → hard cut."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXY"  # 25 chars
        words = _ws([(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate(chars)])
        seg = SubtitleSegmenter(max_chars=15, max_duration=99.0)
        result = seg.segment(words)
        # Must have been split (hard cut) since no punctuation exists
        assert len(result) >= 2
        full_text = "".join(s.text for s in result)
        assert full_text == chars

    def test_no_split_within_limits(self) -> None:
        """Comma is discarded but no split when within limits."""
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("，", 1.0, 1.0), ("世", 1.0, 1.5), ("界", 1.5, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好世界"]

    def test_multiple_commas_splits_when_over_duration(self) -> None:
        words = _ws([
            ("第", 0.0, 1.0), ("一", 1.0, 2.0),
            ("，", 2.0, 2.0),
            ("第", 2.0, 3.5), ("二", 3.5, 5.0),
            ("，", 5.0, 5.0),
            ("第", 5.0, 6.5), ("三", 6.5, 8.5),
        ])
        segs = SubtitleSegmenter(max_duration=5.0).segment(words)
        # Total 8.5s > 5.0; should split at clause positions
        assert len(segs) >= 2


# ---------------------------------------------------------------------------
# TestMergeShort
# ---------------------------------------------------------------------------


class TestMergeShort:
    """Short segments are merged with adjacent segments."""

    def test_short_merged_with_next(self) -> None:
        words = _ws([("你", 0.0, 0.3), ("。", 0.3, 0.3), ("好", 0.3, 1.5)])
        segs = SubtitleSegmenter(min_duration=0.833).segment(words)
        # "你" (0.3s < 0.833) should merge with "好"
        assert _texts(segs) == ["你好"]

    def test_multiple_short_merged(self) -> None:
        words = _ws([
            ("A", 0.0, 0.3), ("。", 0.3, 0.3),
            ("B", 0.3, 0.6), ("。", 0.6, 0.6),
            ("C", 0.6, 1.5),
        ])
        segs = SubtitleSegmenter(min_duration=0.833).segment(words)
        # A (0.3s) and B (0.3s) should merge with C (0.9s)
        assert len(segs) == 1
        assert segs[0].text == "ABC"

    def test_no_merge_if_exceeds_max_chars(self) -> None:
        words = _ws([
            ("ABCDEFGHIJKLMNO", 0.0, 0.3),  # 15 chars, short
            ("。", 0.3, 0.3),
            ("abcdefghijklmno", 0.3, 1.5),  # 15 chars
        ])
        segs = SubtitleSegmenter(max_chars=20, min_duration=0.833).segment(words)
        # Merged would be 30 chars > 20 → no merge
        assert len(segs) == 2

    def test_no_merge_if_exceeds_max_duration(self) -> None:
        words = _ws([
            ("A", 0.0, 0.3),  # short
            ("。", 0.3, 0.3),
            ("B", 0.3, 8.0),  # long
        ])
        segs = SubtitleSegmenter(max_duration=7.0, min_duration=0.833).segment(words)
        # Merged duration 8.0 > 7.0 → no merge
        assert len(segs) == 2


# ---------------------------------------------------------------------------
# TestSegmentationEdgeCases
# ---------------------------------------------------------------------------


class TestSegmentationEdgeCases:
    """Edge cases and real-world scenarios."""

    def test_single_word(self) -> None:
        words = _ws([("好", 0.0, 0.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好"]

    def test_all_punctuation(self) -> None:
        words = _ws([("。", 0.0, 0.0), ("！", 0.0, 0.0), ("？", 0.0, 0.0)])
        segs = SubtitleSegmenter().segment(words)
        assert segs == []

    def test_only_clause_punctuation_between_words(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_segment_timing_from_word_timestamps(self) -> None:
        words = _ws([("你", 1.5, 2.0), ("好", 2.0, 2.8)])
        segs = SubtitleSegmenter().segment(words)
        assert len(segs) == 1
        assert segs[0].start_time == 1.5
        assert segs[0].end_time == 2.8

    def test_realistic_chinese_conversation(self) -> None:
        words = _ws([
            ("我", 0.0, 0.3), ("们", 0.3, 0.6), ("今", 0.6, 0.9), ("天", 0.9, 1.2),
            ("去", 1.2, 1.5), ("了", 1.5, 1.8), ("公", 1.8, 2.1), ("园", 2.1, 2.4),
            ("。", 2.4, 2.4),
            ("天", 2.4, 2.7), ("气", 2.7, 3.0), ("很", 3.0, 3.3), ("好", 3.3, 3.6),
            ("。", 3.6, 3.6),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["我们今天去了公园", "天气很好"]

    def test_default_speaker_id(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert segs[0].speaker_id == "SPEAKER_00"

    def test_stale_clause_idx_after_split(self) -> None:
        """After a clause split trims buf, last_clause_idx must be reset.

        Regression: stale index could cause spurious splits at positions
        where no comma existed in the remaining buffer.
        """
        # max_chars=4, one comma after AB, then CDEFGH triggers char limit
        words = _ws([
            ("A", 0.0, 0.2), ("B", 0.2, 0.4),
            ("，", 0.4, 0.5),
            ("C", 0.5, 0.7), ("D", 0.7, 0.9), ("E", 0.9, 1.1),
            ("F", 1.1, 1.3), ("G", 1.3, 1.5), ("H", 1.5, 1.7),
        ])
        segs = SubtitleSegmenter(max_chars=4).segment(words)
        # First split at comma: "AB" → ok
        # Then "CDEFGH" exceeds max_chars=4, must hard-cut (no comma in buf)
        # Stale idx bug would incorrectly split at index 2 in remaining buf
        for seg in segs:
            assert "，" not in seg.text
