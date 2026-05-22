"""SRT subtitle writer.

Writes pre-segmented TranscriptSegments to SRT format. Does NOT perform
splitting — all segmentation is handled by SubtitleSegmenter upstream.
"""
from __future__ import annotations

from transcribe.data.types import TranscriptSegment


class SrtWriter:
    def __init__(
        self,
        speaker_label: bool = True,
        merge_gap: float = 0.5,
    ) -> None:
        self.speaker_label = speaker_label
        self.merge_gap = merge_gap

    def write(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        speaker_name_map: dict[str, str] | None = None,
    ) -> None:
        """Write segments to an SRT file.

        Segments are sorted by time, adjacent same-speaker segments
        within merge_gap are merged, then written directly.
        SubtitleSegmenter is responsible for all splitting upstream.
        """
        if not segments:
            with open(output_path, "w", encoding="utf-8") as f:
                pass
            return

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        merged = self._merge_adjacent(sorted_segments)
        self._write_file(merged, output_path, speaker_name_map)

    def _merge_adjacent(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Merge adjacent same-speaker segments if gap < merge_gap."""
        if not segments:
            return []
        merged: list[TranscriptSegment] = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if (
                seg.speaker_id == last.speaker_id
                and seg.is_overlap == last.is_overlap
                and (seg.start_time - last.end_time) < self.merge_gap
            ):
                merged[-1] = TranscriptSegment(
                    speaker_id=last.speaker_id,
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                    text=last.text + seg.text,
                    is_overlap=last.is_overlap,
                )
            else:
                merged.append(seg)
        return merged

    def _write_file(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        speaker_name_map: dict[str, str] | None,
    ) -> None:
        lines: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            text = seg.text
            if self.speaker_label:
                display_id = (
                    speaker_name_map.get(seg.speaker_id, seg.speaker_id)
                    if speaker_name_map
                    else seg.speaker_id
                )
                text = f"[{self._speaker_label(display_id)}] {text}"
            lines.append(str(idx))
            lines.append(
                f"{self._format_timestamp(seg.start_time)} --> "
                f"{self._format_timestamp(seg.end_time)}"
            )
            lines.append(text)
            lines.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _speaker_label(speaker_id: str) -> str:
        if speaker_id.startswith("SPEAKER_"):
            try:
                num = int(speaker_id.split("_", 1)[1]) + 1
                return f"说话人{num}"
            except (ValueError, IndexError):
                return speaker_id
        return speaker_id
