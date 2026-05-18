"""Tests for transcribe.models.srt_writer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from transcribe.data.types import TranscriptSegment
from transcribe.models.srt_writer import SrtWriter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _read_output(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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
        # 0.9999 → rounds to 1000 ms → next second
        assert SrtWriter._format_timestamp(0.9999) == "00:00:01,000"

    def test_large_value(self) -> None:
        # 2 hours
        assert SrtWriter._format_timestamp(7200.0) == "02:00:00,000"


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
        # Pass in reverse order
        segments = [
            TranscriptSegment("SPEAKER_00", 5.0, 7.0, "second"),
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "first"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # first should appear before second in the file
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
        # Each SRT entry = index, timestamp, text (3 non-blank lines)
        # indices should be 1, 2, 3
        assert lines[0] == "1"
        assert lines[3] == "2"
        assert lines[6] == "3"


# ------------------------------------------------------------------
# _merge_segments
# ------------------------------------------------------------------

class TestMergeSegments:
    def test_merge_adjacent_same_speaker(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "hello "),
            TranscriptSegment("SPEAKER_00", 2.3, 4.0, "world"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # Should be merged into one entry
        assert "hello world" in content
        assert "00:00:00,000 --> 00:00:04,000" in content

    def test_no_merge_different_speaker(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "hello"),
            TranscriptSegment("SPEAKER_01", 2.1, 4.0, "world"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        # Should remain two entries
        lines = [l for l in content.split("\n") if l.strip()]
        # index 1 and 2 present
        assert "1" in lines[0]
        assert "2" in lines[4]

    def test_no_merge_large_gap(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False, merge_gap=0.5)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "hello"),
            TranscriptSegment("SPEAKER_00", 5.0, 7.0, "world"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        lines = [l for l in content.split("\n") if l.strip()]
        # Each SRT entry = index, timestamp, text (3 non-blank lines)
        assert lines[0] == "1"
        assert lines[3] == "2"

    def test_merge_three_segments(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False, merge_gap=1.0)
        out = str(tmp_path / "out.srt")
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "a"),
            TranscriptSegment("SPEAKER_00", 1.2, 2.0, "b"),
            TranscriptSegment("SPEAKER_00", 2.5, 3.0, "c"),
        ]
        writer.write(segments, out)
        content = _read_output(out)
        assert "abc" in content
        # Only one subtitle entry
        lines = [l for l in content.split("\n") if l.strip()]
        assert lines[0] == "1"
        # No second entry
        assert "2" not in lines


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
