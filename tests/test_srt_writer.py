"""Tests for transcribe.models.srt_writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcribe.data.types import TranscriptSegment, WordTimestamp
from transcribe.models.srt_writer import (
    _CLAUSE_END,
    _REPLACE_SPACE,
    _SENTENCE_END,
    SrtWriter,
    _collapse_spaces,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _read_output(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_srt_entries(content: str) -> list[dict]:
    """Parse SRT content into list of {index, start, end, text}."""
    entries = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            entries.append({
                "index": lines[0],
                "timestamp": lines[1],
                "text": lines[2],
            })
    return entries


# ------------------------------------------------------------------
# _collapse_spaces
# ------------------------------------------------------------------


class TestCollapseSpaces:
    def test_no_spaces(self) -> None:
        assert _collapse_spaces("你好") == "你好"

    def test_multiple_spaces(self) -> None:
        assert _collapse_spaces("你  好") == "你 好"

    def test_leading_trailing(self) -> None:
        assert _collapse_spaces("  你好  ") == "你好"

    def test_all_spaces(self) -> None:
        assert _collapse_spaces("   ") == ""


# ------------------------------------------------------------------
# _format_timestamp
# ------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert SrtWriter._format_timestamp(0.0) == "00:00:00,000"

    def test_61_5(self) -> None:
        assert SrtWriter._format_timestamp(61.5) == "00:01:01,500"

    def test_3661_123(self) -> None:
        assert SrtWriter._format_timestamp(3661.123) == "01:01:01,123"

    def test_rounding(self) -> None:
        assert SrtWriter._format_timestamp(0.9999) == "00:00:01,000"

    def test_large_value(self) -> None:
        assert SrtWriter._format_timestamp(7200.0) == "02:00:00,000"


# ------------------------------------------------------------------
# _speaker_label
# ------------------------------------------------------------------


class TestSpeakerLabel:
    def test_speaker_00(self) -> None:
        assert SrtWriter._speaker_label("SPEAKER_00") == "说话人1"

    def test_speaker_01(self) -> None:
        assert SrtWriter._speaker_label("SPEAKER_01") == "说话人2"

    def test_unknown_id(self) -> None:
        assert SrtWriter._speaker_label("unknown") == "unknown"


# ------------------------------------------------------------------
# Punctuation splitting
# ------------------------------------------------------------------


class TestPunctuationSplitting:
    """Test Pass 1: split-and-clean behaviour."""

    def test_sentence_end_splits(self, tmp_path: Path) -> None:
        """Sentence-end punctuation causes hard splits and is discarded."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 3.0, "你好。世界！"),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        texts = [e["text"] for e in entries]
        # Should split into "你好" and "世界"
        assert "你好" in texts
        assert "世界" in texts
        # Punctuation should NOT appear
        assert "。" not in _read_output(out)
        assert "！" not in _read_output(out)

    def test_clause_end_splits(self, tmp_path: Path) -> None:
        """Clause-end punctuation causes soft splits and is discarded."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        # Long enough duration so soft splits survive
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 6.0, "你好，世界，再见"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # Punctuation should NOT appear
        assert "，" not in content
        entries = _parse_srt_entries(content)
        texts = [e["text"] for e in entries]
        assert "你好" in texts
        assert "世界" in texts
        assert "再见" in texts

    def test_ellipsis_as_sentence_end(self, tmp_path: Path) -> None:
        """…… is treated as sentence-end (hard split)."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 4.0, "嗯……好的"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "……" not in content
        entries = _parse_srt_entries(content)
        texts = [e["text"] for e in entries]
        assert "嗯" in texts
        assert "好的" in texts

    def test_dash_as_sentence_end(self, tmp_path: Path) -> None:
        """— (em dash U+2014) is treated as sentence-end."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 4.0, "第一—第二"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # — should not appear in output (it's in _SENTENCE_END)
        assert "—" not in content

    def test_preserved_punctuation(self, tmp_path: Path) -> None:
        """、（）【】《》"" are preserved as-is in output."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        text = "、（）【】《》\u201c\u201d"
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, text),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # All these chars should be preserved
        for ch in "、（）【】《》\u201c\u201d":
            assert ch in content

    def test_replace_space_punctuation(self, tmp_path: Path) -> None:
        """Characters in _REPLACE_SPACE are replaced with spaces."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你'好\u00b7世\u2013界"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # The characters ' · – should not appear; spaces should
        assert "'" not in content
        assert "\u00b7" not in content
        assert "\u2013" not in content

    def test_no_punctuation_in_final_output(self, tmp_path: Path) -> None:
        """After full processing, handled punctuation chars are removed or replaced."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 6.0, "你好。世界，再见！"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        for ch in "。！，":
            assert ch not in content


# ------------------------------------------------------------------
# Duration enforcement
# ------------------------------------------------------------------


class TestDurationEnforcement:
    """Test Pass 2: soft splits merged back when too short."""

    def test_clause_split_merged_when_short(self, tmp_path: Path) -> None:
        """Short clause splits are merged back together.

        The tail chunk is always marked hard, so with '你好，世界，再见'
        we get: [你好 soft] [世界 soft] [再见 hard-tail].
        The two soft chunks merge into one; the hard tail stays separate.
        """
        writer = SrtWriter(speaker_label=False, min_split_duration=10.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好，世界，再见"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        entries = _parse_srt_entries(content)
        # Soft splits merge; hard tail stays → 2 entries
        assert len(entries) == 2
        assert "你好 世界" in entries[0]["text"]
        assert "再见" in entries[1]["text"]

    def test_hard_split_not_merged(self, tmp_path: Path) -> None:
        """Hard splits (sentence-end) are NOT merged even if short."""
        writer = SrtWriter(speaker_label=False, min_split_duration=10.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好。世界"),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        # Hard splits should remain separate even if short
        assert len(entries) == 2

    def test_long_clause_split_kept(self, tmp_path: Path) -> None:
        """Clause splits with sufficient duration are kept separate."""
        writer = SrtWriter(speaker_label=False, min_split_duration=0.5)
        out = str(tmp_path / "out.srt")
        # 10 seconds total, 4 chars → 2.5s per char
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 10.0, "你好，再见"),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        assert len(entries) == 2


# ------------------------------------------------------------------
# Overlap marking
# ------------------------------------------------------------------


class TestOverlapMarking:
    def test_overlap_no_prefix(self, tmp_path: Path) -> None:
        """is_overlap=True no longer adds [重叠] prefix."""
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好", is_overlap=True),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "[重叠]" not in content
        assert "你好" in content

    def test_no_overlap_prefix(self, tmp_path: Path) -> None:
        """is_overlap=False does NOT add [重叠] prefix."""
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好", is_overlap=False),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "[重叠]" not in content

    def test_overlap_with_speaker_label(self, tmp_path: Path) -> None:
        """Overlap segments show speaker label only, no [重叠] prefix."""
        writer = SrtWriter(speaker_label=True)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好", is_overlap=True),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "[说话人1] 你好" in content
        assert "[重叠]" not in content


# ------------------------------------------------------------------
# Merge adjacent
# ------------------------------------------------------------------


class TestMergeAdjacent:
    def test_merge_same_speaker(self, tmp_path: Path) -> None:
        """Adjacent same-speaker segments within merge_gap are merged."""
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好"),
            TranscriptSegment("SPEAKER_00", 2.3, 4.0, "世界"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "你好世界" in content
        assert "00:00:00,000 --> 00:00:04,000" in content

    def test_no_merge_different_speaker(self, tmp_path: Path) -> None:
        """Different speakers are never merged."""
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好"),
            TranscriptSegment("SPEAKER_01", 2.1, 4.0, "世界"),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        assert len(entries) == 2

    def test_no_merge_large_gap(self, tmp_path: Path) -> None:
        """Same speaker but large gap → not merged."""
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好"),
            TranscriptSegment("SPEAKER_00", 5.0, 7.0, "世界"),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        assert len(entries) == 2

    def test_no_merge_different_overlap(self, tmp_path: Path) -> None:
        """Same speaker but different is_overlap → not merged."""
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好", is_overlap=False),
            TranscriptSegment("SPEAKER_00", 2.1, 4.0, "世界", is_overlap=True),
        ]
        writer.write(segments, out)
        entries = _parse_srt_entries(_read_output(out))
        assert len(entries) == 2


# ------------------------------------------------------------------
# Custom speaker names
# ------------------------------------------------------------------


class TestCustomSpeakerNames:
    def test_custom_speaker_names(self, tmp_path: Path) -> None:
        """speaker_name_map overrides display labels."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_01", 1.0, 2.0, "世界"),
        ]
        name_map = {"SPEAKER_00": "张三", "SPEAKER_01": "李四"}
        writer = SrtWriter(speaker_label=True)
        output = str(tmp_path / "out.srt")
        writer.write(segments, output, speaker_name_map=name_map)

        content = _read_output(output)
        assert "[张三]" in content
        assert "[李四]" in content
        assert "说话人" not in content

    def test_partial_name_map(self, tmp_path: Path) -> None:
        """Unmapped speakers get default label."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_01", 1.0, 2.0, "世界"),
        ]
        name_map = {"SPEAKER_00": "张三"}
        writer = SrtWriter(speaker_label=True)
        output = str(tmp_path / "out.srt")
        writer.write(segments, output, speaker_name_map=name_map)

        content = _read_output(output)
        assert "[张三]" in content
        assert "[说话人2]" in content


# ------------------------------------------------------------------
# write — basic output
# ------------------------------------------------------------------


class TestWriteBasic:
    def test_basic_srt_with_speaker_labels(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=True)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.5, "你好世界"),
            TranscriptSegment("SPEAKER_01", 3.0, 5.0, "你好"),
        ]
        writer.write(segments, out)
        content = _read_output(out)

        assert "[说话人1] 你好世界" in content
        assert "[说话人2] 你好" in content
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "00:00:03,000 --> 00:00:05,000" in content

    def test_srt_without_speaker_labels(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.5, "你好世界"),
        ]
        writer.write(segments, out)
        content = _read_output(out)

        assert "说话人" not in content
        assert "你好世界" in content

    def test_empty_segments_produce_empty_file(self, tmp_path: Path) -> None:
        writer = SrtWriter()
        out = str(tmp_path / "out.srt")
        writer.write([], out)
        assert _read_output(out) == ""

    def test_segments_sorted_by_time(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 5.0, 7.0, "second"),
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "first"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        first_pos = content.index("first")
        second_pos = content.index("second")
        assert first_pos < second_pos

    def test_srt_index_numbers(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "a"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "b"),
            TranscriptSegment("SPEAKER_00", 4.0, 5.0, "c"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        lines = [l for l in content.split("\n") if l.strip()]
        assert lines[0] == "1"
        assert lines[3] == "2"
        assert lines[6] == "3"


# ------------------------------------------------------------------
# Word-level timestamp accuracy
# ------------------------------------------------------------------


class TestWordTimestampAccuracy:
    def test_split_uses_word_timestamps(self, tmp_path: Path) -> None:
        """When words are provided, split timestamps match word boundaries."""
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        # Text "AB，CD" — split at comma (clause end)
        # A: 0.0-1.0, B: 1.0-2.0, C: 5.0-6.0, D: 6.0-7.0
        # Uniform interpolation would give: AB → 0.0-2.33, CD → 2.33-7.0 (WRONG)
        # Word timestamps should give: AB → 0.0-2.0, CD → 5.0-7.0 (CORRECT)
        words = [
            WordTimestamp("A", 0.0, 1.0),
            WordTimestamp("B", 1.0, 2.0),
            WordTimestamp("，", 2.0, 5.0),
            WordTimestamp("C", 5.0, 6.0),
            WordTimestamp("D", 6.0, 7.0),
        ]
        segments = [
            TranscriptSegment(
                "SPEAKER_00", 0.0, 7.0, "AB，CD", words=words,
            ),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        entries = _parse_srt_entries(content)
        assert len(entries) == 2
        # First chunk "AB": start=0.0, end should be ~2.0 (end of word B)
        ts0 = entries[0]["timestamp"]
        assert ts0.startswith("00:00:00,000 -->")
        assert "00:00:02,000" in ts0
        # Second chunk "CD": start should be ~5.0 (start of word C), end=7.0
        ts1 = entries[1]["timestamp"]
        assert "00:00:05,000" in ts1
        assert ts1.endswith("00:00:07,000")

    def test_no_words_falls_back_to_uniform(self, tmp_path: Path) -> None:
        """Without words, uniform interpolation is used (backward compat)."""
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 4.0, "AB，CD"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        entries = _parse_srt_entries(content)
        assert len(entries) == 2
        # Uniform: 5 chars total (A B ， C D), tpc = 4.0/5 = 0.8
        # Comma at index 2 → AB ends at char_ends[1] = 1.6s
        ts0 = entries[0]["timestamp"]
        assert ts0.startswith("00:00:00,000 -->")
        assert "00:00:01,600" in ts0
        # CD starts at char_starts[3] = 2.4s
        ts1 = entries[1]["timestamp"]
        assert "00:00:02,400" in ts1
        assert ts1.endswith("00:00:04,000")
