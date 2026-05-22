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
    ) -> None:
        self.speaker_label = speaker_label

    def write(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        speaker_name_map: dict[str, str] | None = None,
    ) -> None:
        """Write segments to an SRT file.

        Segments are sorted by time and written directly.
        SubtitleSegmenter is responsible for all splitting and merging upstream.
        """
        if not segments:
            with open(output_path, "w", encoding="utf-8") as f:
                pass
            return

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        self._write_file(sorted_segments, output_path, speaker_name_map)

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
