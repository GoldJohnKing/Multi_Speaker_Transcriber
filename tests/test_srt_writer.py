"""Tests for transcribe.models.srt_writer — simplified writer (no splitting)."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcribe.data.types import TranscriptSegment
from transcribe.models.srt_writer import SrtWriter


def _read_output(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_srt_entries(content: str) -> list[dict]:
    """Parse SRT content into list of {index, timestamp, text}."""
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


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert SrtWriter._format_timestamp(0.0) == "00:00:00,000"

    def test_61_5(self) -> None:
        assert SrtWriter._format_timestamp(61.5) == "00:01:01,500"

    def test_3661_123(self) -> None:
        assert SrtWriter._format_timestamp(3661.123) == "01:01:01,123"

    def test_rounding(self) -> None:
        assert SrtWriter._format_timestamp(0.9999) == "00:00:01,000"


class TestSpeakerLabel:
    def test_speaker_00(self) -> None:
        assert SrtWriter._speaker_label("SPEAKER_00") == "说话人1"

    def test_speaker_01(self) -> None:
        assert SrtWriter._speaker_label("SPEAKER_01") == "说话人2"

    def test_unknown_id(self) -> None:
        assert SrtWriter._speaker_label("unknown") == "unknown"


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
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 2.5, "你好世界")]
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


class TestCustomSpeakerNames:
    def test_custom_speaker_names(self, tmp_path: Path) -> None:
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



class TestNoSplitting:
    def test_text_written_as_is(self, tmp_path: Path) -> None:
        """SrtWriter does NOT split text — outputs segments unchanged."""
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 3.0, "你好。世界！再见")]
        writer.write(segments, out)
        content = _read_output(out)
        assert "你好。世界！再见" in content

    def test_punctuation_preserved(self, tmp_path: Path) -> None:
        writer = SrtWriter(speaker_label=False)
        out = str(tmp_path / "out.srt")
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 3.0, "你好，世界")]
        writer.write(segments, out)
        content = _read_output(out)
        assert "你好，世界" in content

