"""SRT subtitle writer for transcribed segments.

Converts TranscriptSegment objects into standard SRT subtitle files with
optional speaker labels and segment merging.
"""

from __future__ import annotations

from transcribe.data.types import TranscriptSegment


class SrtWriter:
    """Writes transcript segments to SRT subtitle format.

    Args:
        speaker_label: If True, prefix each line with a speaker label
            like [说话人1].
        max_chars_per_line: Maximum characters per subtitle line (reserved
            for future line-breaking logic).
        min_duration: Minimum duration in seconds for a subtitle entry.
        merge_gap: Maximum gap in seconds between adjacent same-speaker
            segments to trigger merging.
    """

    def __init__(
        self,
        speaker_label: bool = True,
        max_chars_per_line: int = 20,
        min_duration: float = 1.0,
        merge_gap: float = 0.5,
    ) -> None:
        self.speaker_label = speaker_label
        self.max_chars_per_line = max_chars_per_line
        self.min_duration = min_duration
        self.merge_gap = merge_gap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, segments: list[TranscriptSegment], output_path: str) -> None:
        """Write transcript segments to an SRT file.

        Args:
            segments: List of TranscriptSegment objects.
            output_path: Destination file path.
        """
        if not segments:
            with open(output_path, "w", encoding="utf-8") as f:
                pass  # empty file
            return

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        merged = self._merge_segments(sorted_segments)

        lines: list[str] = []
        for idx, seg in enumerate(merged, start=1):
            lines.append(str(idx))
            lines.append(
                f"{self._format_timestamp(seg.start_time)}"
                f" --> "
                f"{self._format_timestamp(seg.end_time)}"
            )
            text = seg.text
            if self.speaker_label:
                text = f"[{self._speaker_label(seg.speaker_id)}] {text}"
            lines.append(text)
            lines.append("")  # blank line between entries

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format *seconds* as ``HH:MM:SS,mmm``."""
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _merge_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Merge adjacent same-speaker segments when the gap is small."""
        if not segments:
            return []

        merged: list[TranscriptSegment] = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if (
                seg.speaker_id == last.speaker_id
                and (seg.start_time - last.end_time) < self.merge_gap
            ):
                # Extend the previous segment
                merged[-1] = TranscriptSegment(
                    speaker_id=last.speaker_id,
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                    text=last.text + seg.text,
                )
            else:
                merged.append(seg)
        return merged

    @staticmethod
    def _speaker_label(speaker_id: str) -> str:
        """Convert a raw speaker id (e.g. SPEAKER_00) to a display label.

        Maps SPEAKER_00 → 说话人1, SPEAKER_01 → 说话人2, etc.
        Falls back to the raw id if the pattern is unexpected.
        """
        if speaker_id.startswith("SPEAKER_"):
            try:
                num = int(speaker_id.split("_", 1)[1]) + 1
                return f"说话人{num}"
            except (ValueError, IndexError):
                return speaker_id
        return speaker_id
