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
    """Sentence-ending punctuation triggers a hard split; ！？ are retained in output."""

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
        assert _texts(segs) == ["好！", "棒"]

    def test_split_at_question(self) -> None:
        words = _ws([("是", 0.0, 0.5), ("吗", 0.5, 1.0), ("？", 1.0, 1.0), ("对", 1.0, 1.5)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["是吗？", "对"]

    def test_multiple_sentence_end(self) -> None:
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0),
            ("世", 1.0, 1.5), ("界", 1.5, 2.0), ("！", 2.0, 2.0),
            ("再", 2.0, 2.5), ("见", 2.5, 3.0),
        ])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好", "世界！", "再见"]

    def test_trailing_sentence_end_no_empty(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["你好"]

    def test_trailing_exclamation_retained(self) -> None:
        """Trailing ！ is retained in the final segment text."""
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！"]

    def test_consecutive_sentence_end_marks(self) -> None:
        """Consecutive ！？ — only the first attaches to preceding content."""
        words = _ws([("好", 0.0, 1.0), ("！", 1.0, 1.0), ("？", 1.0, 1.0), ("棒", 1.0, 2.0)])
        segs = SubtitleSegmenter().segment(words)
        assert _texts(segs) == ["好！", "棒"]

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
            ("一", 1.0, 3.0), ("段", 3.0, 5.0), ("很", 5.0, 6.5), ("长", 6.5, 8.0),
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
    """Short segments are merged with adjacent segments within the same sentence."""

    def test_short_merged_with_next_within_sentence(self) -> None:
        """Short groups from gap splitting (same sentence) are merged."""
        # No sentence-end punctuation — gap split produces two groups,
        # both short, which should be merged back.
        words = _ws([("你", 0.0, 0.3), ("好", 0.3, 1.5)])
        segs = SubtitleSegmenter(min_duration=0.833).segment(words)
        assert _texts(segs) == ["你好"]

    def test_short_not_merged_across_sentence_boundary(self) -> None:
        """Sentence-end boundary prevents merging short segments."""
        words = _ws([("你", 0.0, 0.3), ("。", 0.3, 0.3), ("好", 0.3, 1.5)])
        segs = SubtitleSegmenter(min_duration=0.833).segment(words)
        # "你" and "好" are in different Pass 1 groups (。split) — not merged
        assert _texts(segs) == ["你", "好"]

    def test_multiple_short_merged_within_sentence(self) -> None:
        """Multiple short gap-split groups in the same sentence merge."""
        # Continuous words with small gaps — no sentence-end punctuation
        words = _ws([
            ("A", 0.0, 0.3),
            ("B", 0.3, 0.6),
            ("C", 0.6, 1.5),
        ])
        segs = SubtitleSegmenter(min_duration=0.833).segment(words)
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
        """Regression: stale last_clause_idx must be reset after _check_limits trims buf.

        Sequence: A B ， C D E F G H with max_chars=6
        1. A B → buf grows to 2 chars, no split
        2. ， → last_clause_idx=2, comma discarded
        3. C D E F → buf=[A,B,C,D,E,F], 6 chars → _check_limits splits at
           last_clause_idx=2 → seg="AB", buf=[C,D,E,F]
           After split, last_clause_idx MUST be reset to None.
        4. G H → buf=[C,D,E,F,G,H], 6 chars → _check_limits should do a
           HARD CUT (not clause split) because last_clause_idx is None.
        Without the fix, last_clause_idx=2 would cause a spurious split at
        a non-comma position (splitting C,D from E,F).
        """
        words = _ws([
            ("A", 0.0, 0.3), ("B", 0.3, 0.6),  # 2 chars
            ("，", 0.6, 0.6),                    # comma
            ("C", 0.6, 0.9), ("D", 0.9, 1.2),
            ("E", 1.2, 1.5), ("F", 1.5, 1.8),   # +4 = 6 chars total
            ("G", 1.8, 2.1), ("H", 2.1, 2.4),   # +2 = 6 more
        ])
        segs = SubtitleSegmenter(max_chars=6, min_duration=0.1).segment(words)
        texts = _texts(segs)
        # First segment: AB (clause split at comma)
        assert texts[0] == "AB"
        # Remaining segments: split via hard cut, NOT at stale index
        assert all("，" not in t for t in texts)


# ---------------------------------------------------------------------------
# TestSplitSentenceEnd
# ---------------------------------------------------------------------------


class TestSplitSentenceEnd:
    """Pass 1: split at sentence-end punctuation, discard it, keep commas."""

    def test_no_sentence_end_returns_single_group(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_sentence_end_creates_split(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0), ("世", 1.0, 1.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 2
        assert [w.word for w in groups[0]] == ["你", "好"]
        assert [w.word for w in groups[1]] == ["世"]

    def test_comma_preserved_in_groups(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 1
        assert [w.word for w in groups[0]] == ["你", "，", "好"]

    def test_empty_input(self) -> None:
        seg = SubtitleSegmenter()
        assert seg._split_sentence_end([]) == []


# ---------------------------------------------------------------------------
# TestSplitByGap
# ---------------------------------------------------------------------------


class TestSplitByGap:
    """Pass 2: split at large inter-word gaps."""

    def test_no_gap_returns_single_group(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 1.0, 1.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_hard_gap_always_splits(self) -> None:
        """Gap > 1.0s splits unconditionally."""
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0),
            ("世", 2.5, 3.0), ("界", 3.0, 3.5),
        ])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 2
        assert [w.word for w in groups[0]] == ["你", "好"]
        assert [w.word for w in groups[1]] == ["世", "界"]

    def test_soft_gap_splits_with_min_chars(self) -> None:
        """Gap > 0.5s splits if ≥5 content chars accumulated."""
        words = _ws([
            ("今", 0.0, 0.1), ("天", 0.1, 0.2), ("天", 0.2, 0.3),
            ("气", 0.3, 0.4), ("很", 0.4, 0.5),
            ("好", 1.1, 1.2),
        ])
        seg = SubtitleSegmenter(gap_soft=0.5)
        groups = seg._split_by_gap(words)
        assert len(groups) == 2

    def test_soft_gap_no_split_below_min_chars(self) -> None:
        """Gap > 0.5s does NOT split if <5 content chars accumulated."""
        words = _ws([
            ("你", 0.0, 0.1), ("好", 0.1, 0.2),
            ("世", 0.8, 0.9), ("界", 0.9, 1.0),
        ])
        seg = SubtitleSegmenter(gap_soft=0.5)
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_single_word_returns_single_group(self) -> None:
        words = _ws([("好", 0.0, 0.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_comma_between_words_no_extra_split(self) -> None:
        """Comma (zero-width) between words should not create a false gap."""
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# TestMergeShortGroups
# ---------------------------------------------------------------------------


class TestMergeShortGroups:
    """Pass 3: merge word groups whose content duration < min_duration."""

    def test_short_group_merged_with_next(self) -> None:
        groups = [
            _ws([("你", 0.0, 0.3)]),
            _ws([("好", 0.3, 2.0)]),
        ]
        seg = SubtitleSegmenter(min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 1
        assert merged[0] == groups[0] + groups[1]

    def test_no_merge_when_over_max_chars(self) -> None:
        groups = [
            _ws([("A" * 15, 0.0, 0.3)]),
            _ws([("B" * 15, 0.3, 2.0)]),
        ]
        seg = SubtitleSegmenter(max_chars=20, min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 2

    def test_no_merge_when_over_max_duration(self) -> None:
        groups = [
            _ws([("A", 0.0, 0.3)]),
            _ws([("B", 0.3, 8.0)]),
        ]
        seg = SubtitleSegmenter(max_duration=7.0, min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 2

    def test_chain_of_short_groups_merged(self) -> None:
        groups = [
            _ws([("A", 0.0, 0.3)]),
            _ws([("B", 0.3, 0.6)]),
            _ws([("C", 0.6, 2.0)]),
        ]
        seg = SubtitleSegmenter(min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# TestSplitOversized
# ---------------------------------------------------------------------------


class TestSplitOversized:
    """Pass 4: split groups exceeding max_duration/max_chars at best point."""

    def test_within_limits_not_split(self) -> None:
        group = _ws([("你", 0.0, 1.0), ("好", 1.0, 2.0)])
        seg = SubtitleSegmenter(max_chars=10, max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 1

    def test_over_duration_splits_at_comma(self) -> None:
        """Comma position gets scoring bonus → preferred split point."""
        group = _ws([
            ("这", 0.0, 0.5), ("是", 0.5, 1.0),
            ("，", 1.0, 1.0),
            ("一", 1.0, 3.0), ("段", 3.0, 5.0), ("很", 5.0, 6.5), ("长", 6.5, 7.5),
        ])
        seg = SubtitleSegmenter(max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 2
        assert SubtitleSegmenter._content_chars(result[0]) == 2
        assert SubtitleSegmenter._content_chars(result[1]) == 4

    def test_over_chars_splits_at_midpoint_when_no_punct(self) -> None:
        """No punctuation → split near midpoint."""
        group = _ws([(f"w{i}", float(i) * 0.2, float(i) * 0.2 + 0.1) for i in range(30)])
        seg = SubtitleSegmenter(max_chars=15, max_duration=999.0)
        result = seg._split_oversized(group)
        assert len(result) >= 2
        for g in result:
            assert SubtitleSegmenter._content_chars(g) <= 15

    def test_recursive_split_for_very_long(self) -> None:
        """Very long input gets split into multiple compliant groups."""
        group = _ws([(f"c{i}", float(i) * 0.1, float(i) * 0.1 + 0.05) for i in range(60)])
        seg = SubtitleSegmenter(max_chars=10, max_duration=999.0)
        result = seg._split_oversized(group)
        assert len(result) >= 5
        for g in result:
            assert SubtitleSegmenter._content_chars(g) <= 10

    def test_single_word_not_split(self) -> None:
        group = _ws([("好", 0.0, 8.0)])
        seg = SubtitleSegmenter(max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestValidateCps
# ---------------------------------------------------------------------------


class TestValidateCps:
    """Pass 5: re-split groups where CPS exceeds limit."""

    def test_normal_cps_not_split(self) -> None:
        groups = [_ws([("你", 0.0, 1.0), ("好", 1.0, 2.0), ("世", 2.0, 3.0)])]
        seg = SubtitleSegmenter(max_cps=12.0)
        result = seg._validate_cps(groups)
        assert len(result) == 1

    def test_high_cps_split(self) -> None:
        # 12 chars in ~0.875s → CPS ≈ 13.7 > 12, triggers split.
        # After split into two halves of 6 chars each in ~0.425s → CPS ≈ 14.1 ≤ 15.
        words: list[tuple[str, float, float]] = []
        for i in range(12):
            t = i * 0.075
            words.append((chr(65 + i), t, t + 0.05))
        group = _ws(words)
        seg = SubtitleSegmenter(max_cps=12.0, max_chars=20, max_duration=999.0)
        result = seg._validate_cps([group])
        assert len(result) >= 2
        for g in result:
            dur = SubtitleSegmenter._content_duration(g)
            chars_count = SubtitleSegmenter._content_chars(g)
            if dur > 0:
                cps = chars_count / dur
                assert cps <= 15.0

    def test_single_word_high_cps_not_split(self) -> None:
        group = _ws([("好", 0.0, 0.01)])
        seg = SubtitleSegmenter(max_cps=12.0)
        result = seg._validate_cps([group])
        assert len(result) == 1
